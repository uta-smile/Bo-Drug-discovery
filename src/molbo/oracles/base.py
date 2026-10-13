"""Oracle plumbing: canonical-SMILES memoisation, unique-call budgets, traces.

Two caches, deliberately kept apart, because they answer different questions:

*Run memo* (``_run_values``) is what the **budget** is spent against.  A method
that re-proposes a molecule it has already seen this run learns nothing new, so
it is not charged again -- but a molecule it has never seen *is* charged even if
some earlier run already computed the answer, because the method did not know
it.  Budget numbers are only comparable to PMO if they count unique molecules
per run, which is what this layer enforces.

*Value cache* (:class:`ValueCache`, SQLite on disk) is purely a compute saver.
It spans runs, seeds and processes.  For the lookup oracle it is pointless; for
Boltz-2 at ~20 s/ligand it is the difference between 90 GPU-hours and an
afternoon, since 20 seeds over one library overlap heavily.

Subclasses implement :meth:`Oracle._evaluate_batch` and nothing else.
"""

from __future__ import annotations

import abc
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..chem import DEFAULT_OPTS, StandardizeOptions, canonical_smiles_cached

INVALID_PREFIX = "\x00invalid\x00"


class OracleBudgetExceeded(RuntimeError):
    """Raised when a run asks for more unique evaluations than it may have."""


@dataclass
class OracleRecord:
    """One budget-consuming evaluation, in the order the run made them."""

    call: int
    """1-based index among unique calls. This is the x-axis of the AUC curve."""

    smiles: str
    """Canonical SMILES, or a tagged form of the raw string if unparseable."""

    value: float | None
    valid: bool
    source: str
    """How the value was obtained: computed, disk_cache, or invalid."""

    wall_time: float
    """Seconds spent computing. Zero for cache hits."""


class ValueCache:
    """SQLite-backed ``(oracle_key, smiles) -> value`` store, shared across runs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS oracle_values (
                   oracle_key TEXT NOT NULL,
                   smiles     TEXT NOT NULL,
                   value      REAL,
                   status     TEXT NOT NULL,
                   wall_time  REAL,
                   created    REAL,
                   PRIMARY KEY (oracle_key, smiles))"""
        )
        self._conn.commit()

    def get(self, oracle_key: str, smiles: str) -> tuple[float | None, bool] | None:
        """``(value, ok)`` if evaluated before, else ``None``.

        A stored failure is a real cache hit -- re-running a deterministic
        oracle that already failed just burns the same time again.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT value, status FROM oracle_values WHERE oracle_key=? AND smiles=?",
                (oracle_key, smiles),
            ).fetchone()
        if row is None:
            return None
        return (row[0], row[1] == "ok")

    def put(
        self,
        oracle_key: str,
        smiles: str,
        value: float | None,
        ok: bool,
        wall_time: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO oracle_values VALUES (?,?,?,?,?,?)",
                (oracle_key, smiles, value, "ok" if ok else "fail", wall_time, time.time()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM oracle_values").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Oracle(abc.ABC):
    """Base class for every oracle tier in the project.

    Parameters
    ----------
    name:
        Short identifier, used in results files and as the cache namespace.
    budget:
        Maximum number of *unique* molecules this run may evaluate. ``None``
        for unlimited (use when enumerating a library up front).
    cache_path:
        SQLite file for the cross-run value cache. ``None`` disables it.
    invalid_value:
        Score handed back for SMILES RDKit cannot parse. PMO uses ``0.0``.
    count_invalid:
        Whether an unparseable proposal consumes budget. PMO charges for it, so
        that is the default; a generative arm should not be able to dodge the
        budget by emitting garbage.
    on_exhausted:
        ``raise`` (default, loud) or ``stop`` -- return ``None`` for anything
        past the budget and set :attr:`exhausted`.
    """

    def __init__(
        self,
        name: str,
        *,
        budget: int | None = None,
        cache_path: str | Path | None = None,
        invalid_value: float | None = 0.0,
        count_invalid: bool = True,
        on_exhausted: str = "raise",
        opts: StandardizeOptions = DEFAULT_OPTS,
        maximize: bool = True,
    ):
        if on_exhausted not in ("raise", "stop"):
            raise ValueError("on_exhausted must be 'raise' or 'stop'")
        self.name = name
        self.budget = budget
        self.invalid_value = invalid_value
        self.count_invalid = count_invalid
        self.on_exhausted = on_exhausted
        self.opts = opts
        self.maximize = maximize
        self._cache = ValueCache(cache_path) if cache_path is not None else None
        self.reset()

    # ---- identity ---------------------------------------------------------

    @property
    def cache_key(self) -> str:
        """Namespace for the disk cache.

        Includes the standardiser version, so changing how molecules are keyed
        invalidates old entries instead of silently mixing them in.  Subclasses
        whose settings change the returned number (a different protein, a
        different Boltz-2 checkpoint) must extend this.
        """
        return f"{self.name}|{self.opts.key()}"

    def describe(self) -> dict:
        """Provenance blob written next to results."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "cache_key": self.cache_key,
            "budget": self.budget,
            "maximize": self.maximize,
            "invalid_value": self.invalid_value,
            "count_invalid": self.count_invalid,
            "standardizer": self.opts.key(),
        }

    # ---- run state --------------------------------------------------------

    def reset(self) -> None:
        """Clear per-run accounting. The disk cache is untouched."""
        self._run_values: dict[str, float | None] = {}
        self.history: list[OracleRecord] = []
        self.n_queries: int = 0
        self.exhausted: bool = False
        self.compute_seconds: float = 0.0
        self.n_disk_hits: int = 0

    @property
    def n_calls(self) -> int:
        """Unique, budget-consuming evaluations so far. The PMO x-axis."""
        return len(self.history)

    @property
    def remaining(self) -> int | float:
        return float("inf") if self.budget is None else max(0, self.budget - self.n_calls)

    # ---- evaluation -------------------------------------------------------

    @abc.abstractmethod
    def _evaluate_batch(self, smiles: Sequence[str]) -> Sequence[float | None]:
        """Score canonical, RDKit-parseable SMILES.

        Called only with molecules that are new to this run and absent from the
        disk cache.  Return ``None`` for a molecule the oracle cannot score.
        """

    def __call__(self, smiles: str | Iterable[str]):
        """Score one molecule or a batch, returning values in the order given."""
        if isinstance(smiles, str):
            return self._score_many([smiles])[0]
        return self._score_many(list(smiles))

    def _score_many(self, raw: list[str]) -> list[float | None]:
        self.n_queries += len(raw)

        # Key every request. Invalid strings key on themselves so that repeating
        # the same garbage is not charged twice.
        keys: list[str] = []
        valid: list[bool] = []
        for s in raw:
            can = canonical_smiles_cached(s) if isinstance(s, str) else None
            keys.append(can if can is not None else INVALID_PREFIX + repr(s))
            valid.append(can is not None)

        # Which keys are new to this run, in first-seen order?
        novel: list[str] = []
        seen_here: set[str] = set()
        for key, ok in zip(keys, valid):
            if key in self._run_values or key in seen_here:
                continue
            if not ok and not self.count_invalid:
                continue
            seen_here.add(key)
            novel.append(key)

        admitted = self._admit(novel)

        # Resolve the admitted keys: invalid, disk cache, or compute.
        to_compute = [k for k in admitted if _is_valid_key(k)]
        cached_hits: dict[str, tuple[float | None, bool]] = {}
        if self._cache is not None:
            still: list[str] = []
            for k in to_compute:
                hit = self._cache.get(self.cache_key, k)
                if hit is None:
                    still.append(k)
                else:
                    cached_hits[k] = hit
            to_compute = still

        computed: dict[str, tuple[float | None, float]] = {}
        if to_compute:
            t0 = time.perf_counter()
            values = list(self._evaluate_batch(to_compute))
            elapsed = time.perf_counter() - t0
            if len(values) != len(to_compute):
                raise RuntimeError(
                    f"{type(self).__name__}._evaluate_batch returned {len(values)} "
                    f"values for {len(to_compute)} molecules"
                )
            per = elapsed / len(to_compute)
            self.compute_seconds += elapsed
            for k, v in zip(to_compute, values):
                computed[k] = (v, per)
                if self._cache is not None:
                    self._cache.put(self.cache_key, k, v, v is not None, per)

        # Record admitted keys in proposal order -- this ordering is exactly
        # what the AUC-versus-calls curve is built from.
        for k in admitted:
            if not _is_valid_key(k):
                self._run_values[k] = self.invalid_value
                self._record(k, self.invalid_value, False, "invalid", 0.0)
            elif k in cached_hits:
                v, ok = cached_hits[k]
                self.n_disk_hits += 1
                self._run_values[k] = v if ok else None
                self._record(k, self._run_values[k], ok, "disk_cache", 0.0)
            else:
                v, dt = computed[k]
                self._run_values[k] = v
                self._record(k, v, v is not None, "computed", dt)

        # Anything not admitted (budget ran out under on_exhausted="stop")
        # resolves to None rather than a score.
        return [
            self._run_values.get(k, None if ok else self.invalid_value)
            for k, ok in zip(keys, valid)
        ]

    def _record(self, smiles: str, value: float | None, valid: bool, source: str, dt: float) -> None:
        self.history.append(OracleRecord(len(self.history) + 1, smiles, value, valid, source, dt))

    def _admit(self, novel: list[str]) -> list[str]:
        """Apply the budget to a list of never-before-seen keys."""
        if self.budget is None:
            return novel
        room = self.budget - self.n_calls
        if len(novel) <= room:
            return novel
        if self.on_exhausted == "raise":
            raise OracleBudgetExceeded(
                f"{self.name}: budget {self.budget} exhausted "
                f"({self.n_calls} unique calls used, {len(novel)} new molecules requested)"
            )
        self.exhausted = True
        return novel[: max(0, room)]

    # ---- reporting --------------------------------------------------------

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        scored = [(r.smiles, r.value) for r in self.history if r.value is not None]
        scored.sort(key=lambda kv: kv[1], reverse=self.maximize)
        return scored[:k]

    def buffer(self) -> dict[str, tuple[float, int]]:
        """``{smiles: (value, call_index)}`` -- the shape PMO's scorer expects."""
        return {
            r.smiles: (float(r.value), r.call)
            for r in self.history
            if r.value is not None
        }

    def to_records(self) -> list[dict]:
        return [asdict(r) for r in self.history]

    def save_trace(self, path: str | Path, extra: dict | None = None) -> Path:
        """Write the run trace plus provenance as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "oracle": self.describe(),
            "n_calls": self.n_calls,
            "n_queries": self.n_queries,
            "n_disk_hits": self.n_disk_hits,
            "compute_seconds": round(self.compute_seconds, 4),
            "exhausted": self.exhausted,
            "history": self.to_records(),
        }
        if extra:
            payload.update(extra)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _is_valid_key(key: str) -> bool:
    return not key.startswith(INVALID_PREFIX)

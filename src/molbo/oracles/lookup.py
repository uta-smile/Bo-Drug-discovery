"""Primary oracle: measured pIC50 by dictionary lookup over an enumerated series.

This is the tier the method gets developed on.  It is free, exact,
deterministic and instant, which is the only way to afford the 20+ seeds per
arm needed before a difference between two BO curves means anything.  It is
also the only tier with real ground truth: Boltz-2 returns a prediction, this
returns what the assay actually measured.

Because the library is enumerated up front, :attr:`LookupOracle.library` doubles
as the discrete search space, and :attr:`LookupOracle.best_value` gives the true
optimum -- so simple regret is available here and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..chem import DEFAULT_OPTS, StandardizeOptions, canonical_smiles
from .base import Oracle


class LookupOracle(Oracle):
    """Dictionary lookup over ``{canonical_smiles: value}``.

    Parameters
    ----------
    table:
        Mapping from SMILES to measured value. Keys are re-canonicalised on
        load with the oracle's own standardiser, so a table built under
        different settings cannot silently half-match.
    on_missing:
        What to do when asked for a molecule outside the library.

        ``none`` (default)
            Return ``None``. The molecule still consumes budget -- proposing an
            unmakeable compound is not free in a real campaign either.
        ``error``
            Raise. Use when the optimiser is supposed to be constrained to the
            library and you want a violation to be loud.
        ``value``
            Return :paramref:`missing_value`.
    """

    def __init__(
        self,
        table: Mapping[str, float],
        *,
        name: str = "lookup",
        on_missing: str = "none",
        missing_value: float | None = None,
        opts: StandardizeOptions = DEFAULT_OPTS,
        maximize: bool = True,
        metadata: dict | None = None,
        assume_canonical: bool = False,
        **kwargs,
    ):
        if on_missing not in ("none", "error", "value"):
            raise ValueError("on_missing must be 'none', 'error' or 'value'")
        self.on_missing = on_missing
        self.missing_value = missing_value
        self.metadata = metadata or {}

        if assume_canonical:
            # Only safe when provenance proves the same standardiser built the
            # table; :meth:`from_dataset` checks that before setting this.
            # Canonicalising 2k molecules costs ~0.6 s, which is nothing once
            # but dominates a 20-seed x 4-arm sweep that rebuilds the oracle.
            self._table = {k: float(v) for k, v in table.items()}
            self.n_collisions = 0
        else:
            self._table = {}
            collisions = 0
            for smi, val in table.items():
                can = canonical_smiles(smi, opts)
                if can is None:
                    continue
                if can in self._table and self._table[can] != float(val):
                    collisions += 1
                self._table[can] = float(val)
            self.n_collisions = collisions
        if not self._table:
            raise ValueError("lookup table is empty after canonicalisation")

        # A lookup is free and exact, so a disk cache would only add IO.
        kwargs.setdefault("cache_path", None)
        super().__init__(name, opts=opts, maximize=maximize, **kwargs)

    # ---- library ----------------------------------------------------------

    @property
    def library(self) -> list[str]:
        """The enumerated search space, in a fixed order."""
        return sorted(self._table)

    @property
    def values(self) -> list[float]:
        """Measured values aligned with :attr:`library`."""
        return [self._table[s] for s in self.library]

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, smiles: str) -> bool:
        can = canonical_smiles(smiles, self.opts)
        return can is not None and can in self._table

    @property
    def best_value(self) -> float:
        """True optimum over the library -- the reference for simple regret."""
        return max(self._table.values()) if self.maximize else min(self._table.values())

    @property
    def best_smiles(self) -> str:
        fn = max if self.maximize else min
        return fn(self._table, key=self._table.__getitem__)

    def simple_regret(self) -> float | None:
        """Gap between the best value found so far and the true optimum."""
        found = [r.value for r in self.history if r.value is not None]
        if not found:
            return None
        best = max(found) if self.maximize else min(found)
        return abs(self.best_value - best)

    def top_fraction_found(self, frac: float = 0.01) -> float:
        """Recall of the true top-``frac`` of the library. A hit-rate style metric."""
        n_top = max(1, int(round(frac * len(self._table))))
        ordered = sorted(self._table.items(), key=lambda kv: kv[1], reverse=self.maximize)
        truth = {s for s, _ in ordered[:n_top]}
        found = {r.smiles for r in self.history if r.value is not None}
        return len(truth & found) / n_top

    # ---- Oracle ------------------------------------------------------------

    @property
    def cache_key(self) -> str:
        return f"{super().cache_key}|n={len(self._table)}"

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            library_size=len(self._table),
            best_value=self.best_value,
            best_smiles=self.best_smiles,
            on_missing=self.on_missing,
            dataset=self.metadata,
        )
        return d

    def _evaluate_batch(self, smiles: Sequence[str]) -> Sequence[float | None]:
        out: list[float | None] = []
        for s in smiles:
            if s in self._table:
                out.append(self._table[s])
            elif self.on_missing == "error":
                raise KeyError(
                    f"{self.name}: {s} is not in the enumerated library "
                    f"({len(self._table)} compounds). "
                    "Constrain the optimiser to LookupOracle.library, or set "
                    "on_missing='none'."
                )
            elif self.on_missing == "value":
                out.append(self.missing_value)
            else:
                out.append(None)
        return out

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        path: str | Path,
        *,
        value_column: str = "pchembl_value",
        smiles_column: str = "canonical_smiles",
        min_heavy_atoms: int | None = None,
        max_heavy_atoms: int | None = None,
        **kwargs,
    ) -> "LookupOracle":
        """Load a series built by ``scripts/build_chembl_series.py``.

        Expects ``<stem>.csv`` and, if present, a sibling ``<stem>.meta.json``
        holding the provenance recorded at build time.
        """
        import pandas as pd

        path = Path(path)
        df = pd.read_csv(path)
        missing = {smiles_column, value_column} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

        meta_path = path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = path.parent / (path.stem + ".meta.json")
        metadata = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )

        n_before = len(df)
        if min_heavy_atoms is not None or max_heavy_atoms is not None:
            # Size-filter the library. Very small fragments and very large
            # molecules are not synthetically comparable members of a
            # hit-to-lead series, and both distort the surrogate: tiny
            # fragments have sparse fingerprints that look dissimilar to
            # everything, and outsized molecules drag the descriptor ranges a
            # constraint arm has to reason over.
            from ..chem import to_mol

            keep = []
            for smi in df[smiles_column]:
                mol = to_mol(smi)
                if mol is None:
                    keep.append(False)
                    continue
                n = mol.GetNumHeavyAtoms()
                keep.append(
                    (min_heavy_atoms is None or n >= min_heavy_atoms)
                    and (max_heavy_atoms is None or n <= max_heavy_atoms)
                )
            df = df[keep]
            if df.empty:
                raise ValueError(
                    f"heavy-atom filter [{min_heavy_atoms}, {max_heavy_atoms}] "
                    f"removed all {n_before} compounds in {path}"
                )

        table = dict(zip(df[smiles_column], df[value_column]))
        kwargs.setdefault("name", f"lookup:{path.stem}")
        metadata = dict(metadata)
        metadata.update(
            heavy_atom_filter=[min_heavy_atoms, max_heavy_atoms],
            n_before_filter=n_before,
            n_after_filter=len(df),
        )

        # Trust the file's keys only if it says it was built the same way.
        opts = kwargs.get("opts", DEFAULT_OPTS)
        kwargs.setdefault(
            "assume_canonical", metadata.get("standardizer") == opts.key()
        )
        return cls(table, metadata=metadata, **kwargs)

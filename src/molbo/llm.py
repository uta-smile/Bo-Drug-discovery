"""The arm under test: an LLM proposes a constraint from the SAR observed so far.

The model never sees the library's unobserved values and never picks a
molecule.  It sees the ``(smiles, value)`` pairs the campaign has already
bought and returns a *hypothesis* -- SMARTS patterns to require or forbid, and
descriptor ranges -- which :mod:`molbo.constraints` compiles to a mask over the
library.  Keeping the interface that narrow is what makes the claim testable:
the LLM's contribution is one object, it is inspectable, and it can be replaced
by a random or oracle-derived constraint to see what it was actually worth.

Backends
--------
Two, behind one interface, because they have very different reproducibility
properties and the project needs to be honest about which one produced a table.

:class:`LlamaCppBackend` (default) talks to a local ``llama serve``.  It is the
better scientific instrument here: greedy decoding (``temperature=0``,
``top_k=1``) with a fixed ``seed`` makes the model itself deterministic, and
llama.cpp compiles the JSON schema to a GBNF grammar, so a schema-valid
response is *guaranteed by construction* rather than hoped for.  It is also
free, which matters when a sweep re-proposes a constraint every few rounds.

:class:`AnthropicBackend` talks to the Claude API.  Claude Opus 5 removed
``temperature``/``top_p``/``top_k`` -- sending them is a 400 -- so there is no
"set temperature to zero" option and the model is *not* reproducible on its
own.

Either way every response is written to disk keyed by a hash of the exact
request *including the backend's sampling settings*, and a repeated request is
a file read.  For llama.cpp the cache is an optimisation; for Anthropic it is
the only thing making a rerun reproducible at all.  Re-running a sweep re-uses
recorded constraints; changing the prompt changes the hash and is visibly a new
experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fragments import Fragment, build_menu, menu_digest, render_menu
from .constraints import (
    DESCRIPTORS,
    AllOf,
    Constraint,
    ConstraintError,
    PropertyConstraint,
    SubstructureConstraint,
)

ANTHROPIC_MODEL = "claude-opus-5"
LLAMACPP_URL = "http://127.0.0.1:8080"

PROMPT_VERSION = "v3"
"""Bump when the prompt or schema changes. Part of the cache key, so old
responses are not silently reused under a new prompt.

v2: descriptor bounds became plain required numbers. Nullable union types
(``{"type": ["number", "null"]}``) are a portability hazard -- llama.cpp
compiles the schema to a GBNF grammar and unions are the first thing to break
there -- and an unbounded side is expressible as a very wide bound anyway.

v3: substructures are chosen from a mined menu by index rather than written as
SMARTS. See :mod:`molbo.fragments` for why -- in short, a 4B model emits
invalid SMARTS essentially every time, which measures its typing rather than
its chemistry.
"""

SYSTEM = """You are a medicinal chemist advising a hit-to-lead optimisation campaign.

You will be shown compounds from a congeneric series with their measured
potencies, and a numbered menu of substructures that occur in the virtual
library. Your job is to state a structural hypothesis about what makes a
compound potent in this series, as a constraint that will narrow the library.

You choose substructures by their menu index. You do not write SMARTS.

Guidance:
- Aim to keep roughly 5-25% of the library. Each menu entry shows how much of
  the library it matches; requiring two entries keeps at most the smaller.
- Prefer one or two well-chosen entries. Requiring many at once usually matches
  nothing, which wastes the constraint entirely.
- Only state a descriptor range if the data supports it. Use a wide range when
  you mean "unbounded on this side".
- You are choosing what to *explore*, not predicting individual values. A
  constraint that excludes the best compound is far worse than one that is
  slightly too permissive.

Respond with JSON only."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {
            "type": "string",
            "description": "One or two sentences of SAR reasoning behind the constraint.",
        },
        "require_fragments": {
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": 3,
            "description": "Menu indices a compound must contain (all of them).",
        },
        "forbid_fragments": {
            "type": "array",
            "items": {"type": "integer"},
            "maxItems": 3,
            "description": "Menu indices that disqualify a compound.",
        },
        "property_bounds": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "descriptor": {"type": "string", "enum": sorted(DESCRIPTORS)},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["descriptor", "min", "max"],
                "additionalProperties": False,
            },
        },
        "confidence": {
            "type": "number",
            "description": "0-1. How strongly the observed data supports this hypothesis.",
        },
    },
    "required": [
        "hypothesis",
        "require_fragments",
        "forbid_fragments",
        "property_bounds",
        "confidence",
    ],
    "additionalProperties": False,
}


class LLMUnavailable(RuntimeError):
    """No cached response and no way to reach a model."""


# ---- caching ---------------------------------------------------------------


@dataclass
class ResponseCache:
    """One JSON file per request hash.

    Files rather than SQLite because these are the artifacts a reviewer will
    want to read: each one holds the prompt, the model's hypothesis and the
    parsed constraint, so the arm's behaviour is auditable without running
    anything.
    """

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - corrupt file
            return None

    def put(self, key: str, payload: dict) -> None:
        self.path(key).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(list(self.root.glob("*.json")))


def request_key(backend_key: str, system: str, prompt: str, schema: dict, version: str) -> str:
    """Hash of everything that could change the answer.

    ``backend_key`` carries the model *and* its sampling settings, so a table
    produced by greedy local decoding can never be confused with one produced
    at a different temperature or by a different model.
    """
    blob = json.dumps(
        {"backend": backend_key, "system": system, "prompt": prompt,
         "schema": schema, "v": version},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# ---- backends --------------------------------------------------------------


class Backend:
    """Something that turns (system, prompt, schema) into a JSON object."""

    def key(self) -> str:
        raise NotImplementedError

    def complete(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        """Return ``(parsed_json, metadata)``."""
        raise NotImplementedError


@dataclass
class LlamaCppBackend(Backend):
    """A local ``llama serve`` over its OpenAI-compatible endpoint.

    Deterministic by construction: ``temperature=0`` with ``top_k=1`` is greedy
    decoding, and ``seed`` pins what little randomness remains. ``response_format``
    with a JSON schema is compiled by llama.cpp into a GBNF grammar that the
    sampler is constrained to, so malformed JSON is not a failure mode -- only
    *semantically* wrong content is (a valid-JSON but invalid SMARTS string),
    which :func:`parse_response` handles by dropping and recording the clause.

    ``enable_thinking=False`` is passed through ``chat_template_kwargs`` because
    Qwen3.5 is a hybrid reasoning model; left on, it spends its budget thinking
    and the server has to be told to stop before it emits the JSON.
    """

    base_url: str = LLAMACPP_URL
    model: str = "qwen3.5-4b"
    temperature: float = 0.0
    top_k: int = 1
    seed: int = 0
    max_tokens: int = 1200
    timeout: int = 600
    api_key: str | None = None
    enable_thinking: bool = False

    def key(self) -> str:
        return (
            f"llamacpp:{self.model}:t={self.temperature}:topk={self.top_k}"
            f":seed={self.seed}:think={int(self.enable_thinking)}"
        )

    @property
    def is_deterministic(self) -> bool:
        return self.temperature == 0.0 and self.top_k == 1

    def health(self) -> tuple[bool, str]:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200, r.read().decode()[:200]
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, str(exc)

    def complete(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "constraint", "strict": True, "schema": schema},
            },
            "temperature": self.temperature,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.URLError as exc:
            raise LLMUnavailable(
                f"could not reach llama.cpp at {self.base_url}: {exc}. "
                "Start it with: llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080"
            ) from exc

        choice = payload["choices"][0]
        text = choice["message"]["content"]
        if not text or not text.strip():
            raise LLMUnavailable(
                f"empty response (finish_reason={choice.get('finish_reason')}); "
                "the model may have spent its budget on reasoning -- "
                "start the server with --reasoning-budget 0"
            )
        return json.loads(text), {
            "usage": payload.get("usage", {}),
            "finish_reason": choice.get("finish_reason"),
        }


@dataclass
class AnthropicBackend(Backend):
    """The Claude API.

    No sampling parameters are sent: Claude Opus 5 removed
    ``temperature``/``top_p``/``top_k`` and rejects them with a 400. The model
    is therefore not reproducible on its own and the response cache is the only
    thing that makes a rerun deterministic.
    """

    model: str = ANTHROPIC_MODEL
    max_tokens: int = 4000

    def key(self) -> str:
        return f"anthropic:{self.model}:sampling=unavailable"

    @property
    def is_deterministic(self) -> bool:
        return False

    def complete(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable(
                "the anthropic package is not installed; pip install anthropic, "
                "or use LlamaCppBackend, or run offline against a warm cache"
            ) from exc

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if message.stop_reason == "refusal":  # pragma: no cover - policy path
            raise LLMUnavailable(
                f"the model declined this request "
                f"({getattr(message.stop_details, 'category', None)})"
            )
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:  # pragma: no cover
            raise LLMUnavailable("no text block in the response")
        return json.loads(text), {
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
        }


# ---- prompt ----------------------------------------------------------------


def format_observations(
    observations: Sequence[tuple[str, float]],
    max_show: int = 40,
    units: str = "pIC50 (higher is more potent)",
) -> str:
    """Render observed SAR as a compact table, most potent first.

    Only the extremes are shown when there are many: the top and bottom of the
    series carry nearly all the SAR signal, and padding the prompt with the
    undifferentiated middle costs tokens without adding information.
    """
    if not observations:
        return "No compounds have been measured yet."

    ordered = sorted(observations, key=lambda kv: kv[1], reverse=True)
    n = len(ordered)
    if n <= max_show:
        shown = [("", s, v) for s, v in ordered]
    else:
        half = max_show // 2
        shown = [("", s, v) for s, v in ordered[:half]]
        shown.append(("...", f"({n - max_show} compounds omitted from the middle)", float("nan")))
        shown += [("", s, v) for s, v in ordered[-half:]]

    lines = [f"{n} compounds measured so far. Values are {units}.", ""]
    for marker, smi, val in shown:
        if marker == "...":
            lines.append(f"  {smi}")
        else:
            lines.append(f"  {val:6.2f}   {smi}")
    return "\n".join(lines)


def build_prompt(
    observations: Sequence[tuple[str, float]],
    library_size: int,
    units: str = "pIC50 (higher is more potent)",
    menu: Sequence[Fragment] = (),
) -> str:
    return (
        f"{format_observations(observations, units=units)}\n\n"
        f"The virtual library to be narrowed contains {library_size} compounds "
        "from the same series.\n\n"
        f"{render_menu(menu, library_size)}\n\n"
        "Propose a constraint that keeps the region of the library most likely "
        "to contain more potent compounds."
    )


# ---- parsing ---------------------------------------------------------------


@dataclass
class ParsedConstraint:
    """A model response turned into a constraint, plus what had to be discarded."""

    constraint: Constraint | None
    hypothesis: str
    confidence: float
    dropped: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    chosen: list[str] = field(default_factory=list)
    """Human-readable menu picks, for the run log."""


def parse_response(payload: dict, menu: Sequence[Fragment] = ()) -> ParsedConstraint:
    """Build a constraint from a model response, dropping unusable clauses.

    Invalid SMARTS is dropped and recorded rather than raised. A campaign that
    crashed because the model emitted a bad pattern would report nothing at
    all, whereas a recorded drop rate is itself a result -- "the model proposed
    an unparseable pattern in N of 20 runs" belongs in the paper. If every
    clause is unusable the constraint is ``None`` and the arm runs
    unconstrained for that round, which is logged upstream as mode ``none``.

    ``None`` bounds are still accepted even though the v2 schema asks for plain
    numbers, so responses cached under v1 stay readable.
    """
    dropped: list[str] = []
    parts: list[Constraint] = []

    require, forbid = [], []
    chosen: list[str] = []
    by_index = {f.index: f for f in menu}

    # Menu indices. An out-of-range index is dropped and recorded rather than
    # clamped: silently snapping it to a neighbour would credit the model with
    # a choice it did not make.
    for field_name, bucket in (("require_fragments", require), ("forbid_fragments", forbid)):
        for raw in payload.get(field_name) or []:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                dropped.append(f"{field_name}:{raw!r}")
                continue
            frag = by_index.get(idx)
            if frag is None:
                dropped.append(f"{field_name}:index_{idx}_out_of_range")
                continue
            bucket.append(frag.smarts)
            chosen.append(f"{field_name.split('_')[0]}[{idx}] {frag.label}")

    # Free-form SMARTS, still accepted for backends that can be trusted with it.
    for field_name, bucket in (("require_smarts", require), ("forbid_smarts", forbid)):
        for smarts in payload.get(field_name) or []:
            try:
                SubstructureConstraint(require=(smarts,))  # compiles, or raises
                bucket.append(smarts)
            except ConstraintError:
                dropped.append(f"{field_name}:{smarts}")
    if require or forbid:
        parts.append(SubstructureConstraint(require=tuple(require), forbid=tuple(forbid)))

    bounds: dict[str, tuple[float | None, float | None]] = {}
    for entry in payload.get("property_bounds") or []:
        name = entry.get("descriptor")
        if name not in DESCRIPTORS:
            dropped.append(f"descriptor:{name}")
            continue
        lo, hi = entry.get("min"), entry.get("max")
        if lo is None and hi is None:
            dropped.append(f"descriptor:{name}:empty")
            continue
        if lo is not None and hi is not None and lo > hi:
            dropped.append(f"descriptor:{name}:inverted")
            continue
        bounds[name] = (lo, hi)
    if bounds:
        parts.append(PropertyConstraint(bounds=bounds))

    constraint: Constraint | None
    if not parts:
        constraint = None
    elif len(parts) == 1:
        constraint = parts[0]
    else:
        constraint = AllOf(parts=tuple(parts))

    return ParsedConstraint(
        constraint=constraint,
        hypothesis=str(payload.get("hypothesis", "")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        dropped=dropped,
        raw=payload,
        chosen=chosen,
    )


# ---- the proposer ----------------------------------------------------------


@dataclass
class LLMConstraintProposer:
    """A :data:`~molbo.campaign.ConstraintSource` backed by a language model.

    Parameters
    ----------
    backend:
        Defaults to a local llama.cpp server -- free, and deterministic under
        greedy decoding.
    cache_dir:
        Where responses are recorded. Re-running a sweep with an unchanged
        prompt makes zero model calls.
    offline:
        Refuse to call the model; raise :class:`LLMUnavailable` on a cache
        miss. Use it to prove a results table was reproduced purely from the
        record.
    units:
        How to describe the objective to the model. The lookup tier is pIC50;
        a PMO task would say what its score means.
    """

    backend: Backend = field(default_factory=LlamaCppBackend)
    cache_dir: str | Path = "data/cache/llm"
    offline: bool = False
    units: str = "pIC50 (higher is more potent)"
    system: str = SYSTEM
    version: str = PROMPT_VERSION
    menu_kwargs: dict = field(default_factory=dict)
    """Passed to :func:`~molbo.fragments.build_menu`. A large, diverse pool needs
    ``mine_sample`` set, or candidate mining walks every molecule."""

    calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._cache = ResponseCache(self.cache_dir)
        self._menu: list[Fragment] | None = None
        self._menu_for: int | None = None

    def menu_for(self, ctx) -> list[Fragment]:
        """Mine the substructure menu once per library and keep it.

        Cached on the proposer rather than rebuilt per call: mining scaffolds
        and ring systems over a 2,000-compound library takes ~13 s, and a
        20-seed sweep would otherwise pay that on every single round.
        """
        if self._menu is None or self._menu_for != id(ctx):
            self._menu = build_menu(ctx.smiles, **self.menu_kwargs)
            self._menu_for = id(ctx)
        return self._menu

    def request_key_for(self, observations, ctx) -> tuple[str, str]:
        """``(cache_key, prompt)`` for a given state, without calling anything.

        Useful for pre-warming a cache offline, and for asserting that a rerun
        really is a cache hit rather than a quiet second call.
        """
        menu = self.menu_for(ctx)
        prompt = build_prompt(
            observations, library_size=len(ctx), units=self.units, menu=menu
        )
        key = request_key(
            self.backend.key(), self.system, prompt, RESPONSE_SCHEMA,
            f"{self.version}|menu={menu_digest(menu)}",
        )
        return key, prompt

    def _complete(self, prompt: str, menu_key: str = "") -> dict:
        """Return the model's JSON, from cache or from the backend."""
        key = request_key(
            self.backend.key(), self.system, prompt, RESPONSE_SCHEMA,
            f"{self.version}|menu={menu_key}",
        )
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit["response"]

        if self.offline:
            raise LLMUnavailable(
                f"offline=True and no cached response for request {key}. "
                "Run once with offline=False to record it."
            )

        response, meta = self.backend.complete(self.system, prompt, RESPONSE_SCHEMA)
        self.calls += 1
        self._cache.put(
            key,
            {
                "backend": self.backend.key(),
                "version": self.version,
                "system": self.system,
                "prompt": prompt,
                "response": response,
                **meta,
            },
        )
        return response

    def __call__(
        self, observations, ctx, rng=None, round_index: int = 0, seed: int = 0, **_
    ) -> Constraint | None:
        menu = self.menu_for(ctx)
        prompt = build_prompt(
            observations, library_size=len(ctx), units=self.units, menu=menu
        )
        try:
            payload = self._complete(prompt, menu_key=menu_digest(menu))
        except LLMUnavailable:
            raise
        except Exception as exc:
            # A malformed response must not kill a 20-seed sweep; the arm runs
            # unconstrained for that round and the failure is counted.
            self.failures += 1
            self.log.append(
                {"round": round_index, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
            return None

        parsed = parse_response(payload, menu=menu)
        self.log.append(
            {
                "round": round_index,
                "seed": seed,
                "n_observations": len(observations),
                "hypothesis": parsed.hypothesis,
                "confidence": parsed.confidence,
                "chosen": parsed.chosen,
                "dropped": parsed.dropped,
                "constraint": parsed.constraint.describe() if parsed.constraint else None,
            }
        )
        return parsed.constraint


@dataclass
class ScriptedProposer:
    """A fixed sequence of responses. For tests and for replaying a recorded run.

    ``menu`` must match the one the responses were produced against, since the
    responses address substructures by index.
    """

    responses: Sequence[dict]
    menu: Sequence[Fragment] = ()
    index: int = field(default=0, init=False)

    def __call__(self, observations, ctx, rng=None, round_index: int = 0, seed: int = 0, **_):
        payload = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return parse_response(payload, menu=self.menu).constraint


@dataclass
class DataDrivenProposer:
    """Non-LLM ablation: a constraint derived from the observed data by statistics.

    Takes the descriptor ranges spanned by the best observed compounds and
    proposes them as bounds. This is the control that asks the sharpest
    question about the whole project: does the model's chemistry knowledge beat
    the trivial thing you would do with the same observations and no model at
    all? If it does not, the LLM is an expensive descriptor summariser.
    """

    top_fraction: float = 0.25
    descriptors: tuple[str, ...] = ("mw", "logp", "tpsa", "hbd", "hba", "aromatic_rings")
    pad: float = 0.1
    """Widen each range by this fraction of its span, so the constraint is a
    region rather than a box drawn exactly through the observed points."""

    def __call__(self, observations, ctx, rng=None, round_index: int = 0, seed: int = 0, **_):
        if len(observations) < 8:
            return None
        ordered = sorted(observations, key=lambda kv: kv[1], reverse=True)
        k = max(4, int(round(self.top_fraction * len(ordered))))
        top = {s for s, _ in ordered[:k]}
        idx = [i for i, s in enumerate(ctx.smiles) if s in top]
        if not idx:
            return None

        bounds: dict[str, tuple[float | None, float | None]] = {}
        for name in self.descriptors:
            vals = ctx.descriptors[name][idx]
            vals = vals[~np.isnan(vals)]
            if vals.size < 3:
                continue
            lo, hi = float(vals.min()), float(vals.max())
            span = max(hi - lo, 1e-6)
            bounds[name] = (lo - self.pad * span, hi + self.pad * span)
        return PropertyConstraint(bounds=bounds) if bounds else None


# ---- selection: BO proposes, the model decides ------------------------------

SELECT_VERSION = "s3"

SELECT_SYSTEM = """You are a medicinal chemist choosing the next compound to send to assay.

You will be shown compounds already measured in this series, then a shortlist of
untested candidates that a Bayesian optimiser ranked highest. Exactly one of them
will be synthesised and tested next. Pick it.

The shortlist is already good by the model's reckoning, so your job is the part
the model is bad at: recognising which candidate is chemically informative, which
looks like a dead end, and which repeats something already learned.

Guidance:
- The model's predicted value and uncertainty are shown. High uncertainty can be
  worth buying if the answer would teach you something; it is not worth buying
  twice.
- Prefer a candidate that resolves a real SAR question over one that is a minor
  variation on a compound already measured.
- You must choose by index, from the shortlist only.

Respond with JSON only."""

SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # `choice` is deliberately first. The grammar emits object keys in
        # schema order, so putting the answer before the prose means a response
        # truncated at max_tokens still carries a usable choice. With reasoning
        # first, 6 of 41 calls in a pilot run overran and lost their answer
        # entirely -- a 17% failure rate that looks, in aggregate, exactly like
        # a model that agrees with BO.
        "choice": {
            "type": "integer",
            "description": "Index of the chosen candidate in the shortlist.",
        },
        "reasoning": {
            "type": "string",
            "maxLength": 400,
            "description": "One or two sentences. Why this candidate, over the others.",
        },
    },
    "required": ["choice", "reasoning"],
    "additionalProperties": False,
}


def build_select_prompt(
    shortlist: Sequence[str],
    mu: Sequence[float],
    sd: Sequence[float],
    observations: Sequence[tuple[str, float]],
    units: str = "pIC50 (higher is more potent)",
    max_show: int = 24,
) -> str:
    lines = [format_observations(observations, max_show=max_show, units=units), ""]
    # An arm with no surrogate (``ArmConfig.strategy == "llm"``) passes NaN
    # here. Showing "pred nan +/- nan" would be noise the model has to ignore,
    # and a placeholder number would be worse -- a fabricated prediction is
    # indistinguishable from a real one inside the prompt. Drop the column
    # instead, and say where the list came from: "ranked by the optimiser" is a
    # false statement about a uniform draw, and that difference is exactly what
    # the no-surrogate arm is measuring.
    #
    # `any`, not `all`: the no-surrogate arm passes NaN for *every* entry, so
    # that case is unambiguous. A BO arm whose GP returned one non-finite value
    # is a numerical fault, not a uniform draw, and `all` would silently relabel
    # its GP-ranked shortlist as "drawn at random" -- a false statement to the
    # model, a different cache key, and nothing in the log to say it happened.
    # Keep the posterior column whenever any of it is real.
    have_posterior = bool(len(shortlist)) and any(
        np.isfinite(m) and np.isfinite(sd_i) for m, sd_i in zip(mu, sd)
    )

    if have_posterior:
        lines.append(
            f"{len(shortlist)} untested candidates, ranked by the optimiser "
            "(pred = predicted value, +/- = model uncertainty):"
        )
        lines.append("")
        for i, (smi, m, sd_i) in enumerate(zip(shortlist, mu, sd)):
            lines.append(f"  [{i:2d}] pred {m:6.2f} +/- {sd_i:4.2f}   {smi}")
    else:
        lines.append(
            f"{len(shortlist)} untested candidates, drawn at random from the "
            "untested pool. There is no model prediction for them:"
        )
        lines.append("")
        for i, smi in enumerate(shortlist):
            lines.append(f"  [{i:2d}] {smi}")

    lines.append("")
    lines.append("Which one should be tested next? Answer with its index.")
    return "\n".join(lines)


@dataclass
class LLMSelector:
    """BO proposes a shortlist; the model decides which one costs an oracle call.

    This is a different bet from :class:`LLMConstraintProposer`, and a safer
    one. A constraint is *absorbing*: one bad mask fences off the optimum for
    the rest of the run and BO cannot recover. A bad pick here costs exactly one
    evaluation, and the next round is a fresh decision.

    It is also far more expensive -- one call per evaluation rather than one per
    run -- and the cache cannot help, because every round presents a state that
    has never occurred before.

    ``ArmConfig.shortlist_k`` is the dial that says how much of the decision the
    model is actually making: at ``k=1`` this arm is exactly plain BO, and at
    ``k = len(candidates)`` BO only supplies the posterior.
    """

    backend: Backend = field(default_factory=LlamaCppBackend)
    cache_dir: str | Path = "data/cache/llm_select"
    offline: bool = False
    units: str = "pIC50 (higher is more potent)"
    system: str = SELECT_SYSTEM
    version: str = SELECT_VERSION
    max_show: int = 24
    max_tokens: int = 700
    shuffle: bool = True
    """Present the shortlist in a seeded random order instead of acquisition order.

    Not cosmetic -- it removes a confound that makes the arm uninterpretable.
    Measured on Qwen3.5-4B, the choice distribution over a 15-long shortlist is
    strongly non-uniform (chi-square vs uniform p < 1e-4) but the shape is
    positional: peaks at the first and last entries, a trough at the second.
    Presented in acquisition order, "picks position 0" and "agrees with BO's
    top candidate" are the same event, so primacy bias masquerades as good
    judgment. Shuffling decouples them: any remaining edge over
    :class:`RandomShortlistSelector` is then attributable to the molecules
    rather than to where they sat on the page."""

    calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    deferred: int = field(default=0, init=False)
    ranks: list[int] = field(default_factory=list, init=False)
    """Acquisition rank of each pick (0 = BO's own top candidate)."""

    shown_positions: list[int] = field(default_factory=list, init=False)
    """Where the pick sat in the list as presented. Compare with ``ranks``:
    a flat ``ranks`` distribution next to a peaked ``shown_positions`` one is
    the signature of a model reading position rather than chemistry."""

    def __post_init__(self):
        self._cache = ResponseCache(self.cache_dir)

    def __call__(
        self, shortlist, scores, mu, sd, observations, ctx=None, rng=None,
        round_index: int = 0, seed: int = 0, q: int = 1, **_
    ):
        # A per-(seed, round) stream, so the ordering is reproducible without
        # perturbing the campaign's own RNG (which would desynchronise this arm
        # from every other one and break the paired comparison).
        order = np.arange(len(shortlist))
        if self.shuffle:
            np.random.default_rng((seed + 1) * 100_003 + round_index).shuffle(order)
        shown = [shortlist[i] for i in order]
        shown_mu = [mu[i] for i in order]
        shown_sd = [sd[i] for i in order]

        prompt = build_select_prompt(
            shown, shown_mu, shown_sd, observations,
            units=self.units, max_show=self.max_show,
        )
        key = request_key(self.backend.key(), self.system, prompt, SELECT_SCHEMA, self.version)

        payload = self._cache.get(key)
        if payload is not None:
            self.cache_hits += 1
            payload = payload["response"]
        elif self.offline:
            raise LLMUnavailable(f"offline=True and no cached response for {key}")
        else:
            try:
                payload, meta = self.backend.complete(
                    self.system, prompt, SELECT_SCHEMA
                )
                self.calls += 1
                self._cache.put(
                    key,
                    {"backend": self.backend.key(), "version": self.version,
                     "system": self.system, "prompt": prompt, "response": payload, **meta},
                )
            except LLMUnavailable:
                raise
            except Exception:
                # Defer to BO's own ranking rather than ending the run. Counted,
                # so a model that fails often cannot look like one that agrees
                # with BO often.
                self.failures += 1
                return None

        choice = payload.get("choice")
        try:
            rank = int(choice)
        except (TypeError, ValueError):
            self.deferred += 1
            return None
        if not (0 <= rank < len(shortlist)):
            self.deferred += 1
            return None
        # Map the position the model saw back to the acquisition rank, so the
        # logged rank always means "how far down BO's ranking did it go".
        true_rank = int(order[rank])
        self.ranks.append(true_rank)
        self.shown_positions.append(rank)
        return true_rank


@dataclass
class RandomShortlistSelector:
    """Control: pick uniformly at random from BO's shortlist.

    Without this arm an LLM selector that beats plain BO proves nothing, because
    *any* deviation from the acquisition argmax adds exploration, and at a small
    budget exploration alone can help. This isolates the model's judgment from
    the mere fact of not always taking rank 0.
    """

    def __call__(self, shortlist, rng=None, q: int = 1, **_):
        r = rng if rng is not None else np.random.default_rng(0)
        return [int(i) for i in r.choice(len(shortlist), size=min(q, len(shortlist)), replace=False)]


# ---- warm start: choosing the initial design --------------------------------

INIT_VERSION = "i1"

INIT_SYSTEM = """You are a medicinal chemist starting a hit-to-lead campaign.

Nothing has been measured yet. You will be shown a random sample of compounds
from the screening library and must choose which ones to assay first.

This is the one point in the campaign where chemical knowledge has nothing to
compete with: there is no data, so a model fitted to the results cannot help.
Choose the compounds whose measurements would teach you the most -- covering the
structural options in the sample rather than repeating one of them, and
favouring chemotypes you believe are plausible for a potent inhibitor.

Respond with JSON only."""


def init_schema(n_choices: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "choices": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": n_choices,
                "maxItems": n_choices,
                "description": f"Exactly {n_choices} distinct indices from the sample.",
            },
            "reasoning": {"type": "string", "maxLength": 400},
        },
        "required": ["choices", "reasoning"],
        "additionalProperties": False,
    }


def build_init_prompt(sample: Sequence[str], n_choices: int, units: str) -> str:
    lines = [
        f"A random sample of {len(sample)} compounds from the screening library:",
        "",
    ]
    lines += [f"  [{i:2d}] {smi}" for i, smi in enumerate(sample)]
    lines += [
        "",
        f"Nothing has been measured yet. The objective is {units}.",
        f"Choose exactly {n_choices} of them to assay first, by index.",
    ]
    return "\n".join(lines)


@dataclass
class LLMInitSelector:
    """Warm start: the model picks the initial design from a random sample.

    The comparison this enables is unusually tight. The sample is drawn with the
    campaign's own seed -- the same draw plain BO would have taken its random
    initial design from -- so the only difference between this arm and plain BO
    is *which* members of an identical candidate pool get bought first. Nothing
    else about the run changes.

    It is also where a language model has the best case it will ever get: at the
    cold start there is no data, so the surrogate has nothing to offer and prior
    knowledge is the only thing on the table. One call per run.
    """

    backend: Backend = field(default_factory=LlamaCppBackend)
    cache_dir: str | Path = "data/cache/llm_init"
    offline: bool = False
    units: str = "pIC50 (higher is more potent)"
    system: str = INIT_SYSTEM
    version: str = INIT_VERSION
    sample_size: int = 40
    max_tokens: int = 700

    calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._cache = ResponseCache(self.cache_dir)

    def __call__(self, library, ctx=None, rng=None, seed: int = 0, n_init: int = 10, **_):
        n = len(library)
        m = min(self.sample_size, n)
        # Same seed, same draw plain BO's initial design comes from.
        pool = np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))
        sample = [library[i] for i in pool]
        schema = init_schema(n_init)
        prompt = build_init_prompt(sample, n_init, self.units)
        key = request_key(self.backend.key(), self.system, prompt, schema, self.version)

        payload = self._cache.get(key)
        if payload is not None:
            self.cache_hits += 1
            payload = payload["response"]
        elif self.offline:
            raise LLMUnavailable(f"offline=True and no cached response for {key}")
        else:
            try:
                payload, meta = self.backend.complete(self.system, prompt, schema)
                self.calls += 1
                self._cache.put(key, {"backend": self.backend.key(), "version": self.version,
                                      "system": self.system, "prompt": prompt,
                                      "response": payload, **meta})
            except LLMUnavailable:
                raise
            except Exception:
                self.failures += 1
                return None  # fall back to the random design

        picked, seen = [], set()
        for raw in payload.get("choices") or []:
            try:
                j = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= j < m and j not in seen:
                seen.add(j)
                picked.append(int(pool[j]))
        # Top up from the same pool if the model under-delivered, so the arm
        # always starts with exactly n_init compounds like every other arm.
        for j in range(m):
            if len(picked) >= n_init:
                break
            if int(pool[j]) not in picked:
                picked.append(int(pool[j]))

        self.log.append({"seed": seed, "n_chosen": len(seen), "sample_size": m,
                         "reasoning": str(payload.get("reasoning", ""))[:300]})
        return picked[:n_init]


@dataclass
class DiverseInitSelector:
    """Non-LLM control for the warm start: MaxMin diversity over the same sample.

    The sharpest question about a warm-start win is *what* the model actually
    contributed. Choosing ten structurally spread-out compounds gives the GP
    better coverage and would help on its own, with no chemistry involved. This
    arm does exactly that -- greedy MaxMin on Tanimoto distance over the same
    seeded 40-molecule sample the model sees -- so if the two arms score alike,
    the win is about diversity and not about knowing any chemistry.
    """

    sample_size: int = 40

    def __call__(self, library, ctx=None, rng=None, seed: int = 0, n_init: int = 10, **_):
        from .surrogate import fingerprint_matrix, tanimoto_gram

        n = len(library)
        m = min(self.sample_size, n)
        pool = np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))
        X = fingerprint_matrix([library[i] for i in pool])
        sim = tanimoto_gram(X, X)

        # Start from the sample's most "central" member so the first pick does
        # not depend on library order, then repeatedly take whatever is
        # furthest from everything chosen so far.
        chosen = [int(np.argmax(sim.sum(axis=1)))]
        while len(chosen) < min(n_init, m):
            worst = sim[:, chosen].max(axis=1)
            worst[chosen] = np.inf
            chosen.append(int(np.argmin(worst)))
        return [int(pool[j]) for j in chosen]

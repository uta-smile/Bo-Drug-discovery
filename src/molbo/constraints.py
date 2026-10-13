"""Machine-checkable constraints over a library -- the object the LLM arm produces.

The hypothesis under test is that an LLM, shown the SAR observed so far, can
narrow the search space usefully.  For that claim to be falsifiable the LLM's
output has to be something a machine can check against every library member
without asking the LLM again, so the arm's interface is deliberately narrow: a
constraint is a JSON spec of SMARTS patterns and descriptor ranges, and it
compiles to a boolean mask.  Nothing here calls a model; see
:mod:`molbo.llm` for the thing that produces the spec and caches it.

Two design rules the experiment depends on:

*Constraints are never allowed to empty the search space.*  A hard mask that
retains nothing turns BO into a crash, and a mask that retains four molecules
turns a 200-call budget into a 4-call one -- either way the arm's curve stops
measuring the thing being studied.  :func:`apply_constraints` therefore falls
back: hard mask, then soft penalty, then no constraint, recording which
happened so the failure rate is reportable rather than invisible.

*Every constraint arm needs its controls.*  An LLM constraint that helps might
be helping because the LLM knows medicinal chemistry, or merely because *any*
restriction of the candidate set helps at this budget.  :class:`RandomSubsetConstraint`
is the control that separates those, and :func:`oracle_constraint` is the
ceiling -- a constraint built from the answers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from .chem import to_mol

# Descriptors an LLM may reference by name in a constraint spec. Kept small and
# standard on purpose: these are the ones a medicinal chemist would actually
# state a rule about, and every one is cheap and deterministic.
DESCRIPTORS = {
    "mw": lambda m: Descriptors.MolWt(m),
    "logp": lambda m: Crippen.MolLogP(m),
    "tpsa": lambda m: rdMolDescriptors.CalcTPSA(m),
    "hbd": lambda m: rdMolDescriptors.CalcNumHBD(m),
    "hba": lambda m: rdMolDescriptors.CalcNumHBA(m),
    "rotb": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
    "rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "aromatic_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "fsp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "formal_charge": lambda m: Chem.GetFormalCharge(m),
    "stereocenters": lambda m: rdMolDescriptors.CalcNumAtomStereoCenters(m),
}


class ConstraintError(ValueError):
    """The spec was malformed -- a bad SMARTS, an unknown descriptor."""


def descriptor_table(smiles: Sequence[str]) -> dict[str, np.ndarray]:
    """Every descriptor for every molecule, computed once per library.

    Worth the up-front cost: a 20-seed sweep re-evaluates constraints hundreds
    of times over the same fixed library, and RDKit descriptor calls dominate
    everything else in the loop if left inside it.
    """
    mols = [to_mol(s) for s in smiles]
    out: dict[str, np.ndarray] = {}
    for name, fn in DESCRIPTORS.items():
        vals = np.full(len(smiles), np.nan, dtype=float)
        for i, m in enumerate(mols):
            if m is not None:
                try:
                    vals[i] = float(fn(m))
                except Exception:  # pragma: no cover - descriptor edge cases
                    pass
        out[name] = vals
    return out


@dataclass
class LibraryContext:
    """A library, with its parsed molecules and descriptors computed on demand.

    Both are **lazy and cached**, which is a correctness property rather than an
    optimisation. An earlier version stored them as plain fields, so
    ``LibraryContext(smiles=...)`` produced a context with an empty ``mols``
    list -- and every substructure mask built from it came back all-False. That
    is indistinguishable from "the constraint matched nothing", so
    :func:`apply_constraints` degraded to unconstrained and the arm silently
    became plain BO. It cost a full PMO sweep that returned dAUC = +0.0000 on
    every task before the zero was recognised as a bug rather than a result.

    Computing on first use means a context is correct however it was built, and
    a caller that never touches substructures never pays for the parse.
    """

    smiles: list[str]
    _mols: list[Chem.Mol | None] | None = field(default=None, repr=False)
    _descriptors: dict[str, np.ndarray] | None = field(default=None, repr=False)
    _masks: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def mols(self) -> list[Chem.Mol | None]:
        if self._mols is None:
            self._mols = [to_mol(s) for s in self.smiles]
        return self._mols

    @property
    def descriptors(self) -> dict[str, np.ndarray]:
        if self._descriptors is None:
            out: dict[str, np.ndarray] = {}
            for name, fn in DESCRIPTORS.items():
                vals = np.full(len(self.smiles), np.nan, dtype=float)
                for i, m in enumerate(self.mols):
                    if m is not None:
                        try:
                            vals[i] = float(fn(m))
                        except Exception:  # pragma: no cover
                            pass
                out[name] = vals
            self._descriptors = out
        return self._descriptors

    def mask_for(self, constraint: "Constraint") -> np.ndarray:
        """Cached mask for ``constraint`` over this library.

        A constraint is re-applied every round, but it only *changes* when the
        source proposes a new one -- and a substructure match over a 250k-member
        pool takes the better part of a minute. Without this, a single PMO
        campaign would spend over an hour recomputing a mask that never moved.
        Keyed on the constraint's spec hash, so a genuinely new proposal is
        recomputed and an unchanged one is not.
        """
        if self._masks is None:
            self._masks = {}
        key = constraint.fingerprint()
        cached = self._masks.get(key)
        if cached is None:
            cached = constraint.mask(self)
            self._masks[key] = cached
        return cached

    @classmethod
    def build(cls, smiles: Sequence[str]) -> "LibraryContext":
        """Kept for callers that want the work done up front."""
        ctx = cls(smiles=list(smiles))
        ctx.mols  # noqa: B018 - force the parse now
        return ctx

    def __len__(self) -> int:
        return len(self.smiles)


class Constraint:
    """Base class: something that turns a library into a boolean mask."""

    kind = "constraint"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}

    def fingerprint(self) -> str:
        """Stable hash of the spec, for caching and for results provenance."""
        blob = json.dumps(self.describe(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class SubstructureConstraint(Constraint):
    """Require every pattern in ``require`` and forbid every one in ``forbid``.

    ``require`` is an AND, not an OR: "this series needs the hinge-binding
    quinazoline N1 *and* a para substituent" is the kind of statement a chemist
    makes, and an OR would be satisfied by either half. For an OR, use
    :class:`AnyOf`.
    """

    require: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    kind: str = "substructure"

    def __post_init__(self):
        self._req = [self._compile(p) for p in self.require]
        self._forb = [self._compile(p) for p in self.forbid]

    @staticmethod
    def _compile(smarts: str):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ConstraintError(f"unparseable SMARTS: {smarts!r}")
        return patt

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        out = np.zeros(len(ctx), dtype=bool)
        for i, m in enumerate(ctx.mols):
            if m is None:
                continue
            if all(m.HasSubstructMatch(p) for p in self._req) and not any(
                m.HasSubstructMatch(p) for p in self._forb
            ):
                out[i] = True
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "require": list(self.require), "forbid": list(self.forbid)}


@dataclass
class PropertyConstraint(Constraint):
    """Inclusive ``{descriptor: [lo, hi]}`` ranges; ``None`` for an open end.

    A molecule whose descriptor is NaN (RDKit could not compute it) fails the
    constraint rather than passing it -- an unevaluable molecule is not
    evidence that the rule is satisfied.
    """

    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    kind: str = "property"

    def __post_init__(self):
        unknown = set(self.bounds) - set(DESCRIPTORS)
        if unknown:
            raise ConstraintError(
                f"unknown descriptor(s) {sorted(unknown)}; known: {sorted(DESCRIPTORS)}"
            )

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        out = np.ones(len(ctx), dtype=bool)
        for name, (lo, hi) in self.bounds.items():
            vals = ctx.descriptors[name]
            ok = ~np.isnan(vals)
            if lo is not None:
                ok &= vals >= lo
            if hi is not None:
                ok &= vals <= hi
            out &= ok
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "bounds": {k: list(v) for k, v in self.bounds.items()}}


@dataclass
class AllOf(Constraint):
    """Conjunction. The default way an LLM spec with several clauses combines."""

    parts: tuple[Constraint, ...] = ()
    kind: str = "all_of"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        out = np.ones(len(ctx), dtype=bool)
        for p in self.parts:
            out &= p.mask(ctx)
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "parts": [p.describe() for p in self.parts]}


@dataclass
class AnyOf(Constraint):
    """Disjunction -- "either of these two warheads is fine"."""

    parts: tuple[Constraint, ...] = ()
    kind: str = "any_of"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        if not self.parts:
            return np.ones(len(ctx), dtype=bool)
        out = np.zeros(len(ctx), dtype=bool)
        for p in self.parts:
            out |= p.mask(ctx)
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "parts": [p.describe() for p in self.parts]}


@dataclass
class RandomSubsetConstraint(Constraint):
    """Control arm: keep a random ``fraction`` of the library.

    This is the arm that decides whether the LLM contributed chemistry or just
    contributed *a restriction*.  Narrowing the candidate set changes BO's
    exploration behaviour on its own -- fewer candidates means the acquisition
    argmax is less likely to wander -- so an LLM arm must beat a random mask of
    comparable tightness, not merely beat unconstrained BO.
    """

    fraction: float = 0.25
    seed: int = 0
    kind: str = "random_subset"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = len(ctx)
        k = max(1, int(round(self.fraction * n)))
        out = np.zeros(n, dtype=bool)
        out[rng.choice(n, size=k, replace=False)] = True
        return out

    def describe(self) -> dict:
        return {"kind": self.kind, "fraction": self.fraction, "seed": self.seed}


@dataclass
class MaskConstraint(Constraint):
    """A precomputed mask, used for the ceiling arm and for tests."""

    values: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=bool))
    label: str = "mask"
    kind: str = "explicit_mask"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        if self.values.shape[0] != len(ctx):
            raise ConstraintError(
                f"mask has {self.values.shape[0]} entries for a library of {len(ctx)}"
            )
        return self.values.astype(bool)

    def describe(self) -> dict:
        return {"kind": self.kind, "label": self.label, "n_true": int(self.values.sum())}


def oracle_constraint(values: Sequence[float], fraction: float = 0.25, maximize: bool = True):
    """Ceiling arm: the mask you would write if you already knew the answers.

    Not a method -- a reference line. It bounds how much *any* constraint on
    this library could possibly buy, which is the number that says whether the
    LLM arm has room to be interesting at all.
    """
    vals = np.asarray(values, dtype=float)
    k = max(1, int(round(fraction * vals.size)))
    order = np.argsort(-vals if maximize else vals)
    out = np.zeros(vals.size, dtype=bool)
    out[order[:k]] = True
    return MaskConstraint(values=out, label=f"top_{fraction:g}_by_truth")


# ---- application ----------------------------------------------------------


@dataclass
class ConstraintOutcome:
    """What actually happened when a constraint met the library.

    ``mode`` is the tier that ended up being used: ``hard``, ``soft`` (too
    tight, applied as an acquisition penalty instead) or ``none`` (the mask
    kept nothing at all). Aggregated across a run this is the arm's
    "how often did the LLM propose something unusable" rate, which belongs in
    the paper whichever way the headline result goes.
    """

    mask: np.ndarray
    penalty: np.ndarray
    mode: str
    n_allowed: int
    n_library: int
    constraint: dict

    @property
    def fraction_allowed(self) -> float:
        return self.n_allowed / max(1, self.n_library)


def apply_constraints(
    constraint: Constraint | None,
    ctx: LibraryContext,
    *,
    min_candidates: int = 20,
    soft_penalty: float = 1.0,
    already_seen: np.ndarray | None = None,
) -> ConstraintOutcome:
    """Compile a constraint to a mask, degrading gracefully if it is too tight.

    Parameters
    ----------
    min_candidates:
        Below this many *unseen* survivors the mask is demoted from hard to
        soft. The count is of unseen molecules on purpose: a constraint that
        admits 200 compounds the campaign has already bought is as useless as
        one that admits none.
    soft_penalty:
        Subtracted from the acquisition score of a disallowed molecule when
        the constraint is applied softly, in units of the objective.
    already_seen:
        Boolean mask of library members this run has already evaluated.
    """
    n = len(ctx)
    if constraint is None:
        return ConstraintOutcome(
            np.ones(n, bool), np.zeros(n), "none", n, n, {"kind": "unconstrained"}
        )

    mask = ctx.mask_for(constraint)
    unseen = ~already_seen if already_seen is not None else np.ones(n, bool)
    n_live = int((mask & unseen).sum())
    spec = constraint.describe()

    if n_live >= min_candidates:
        return ConstraintOutcome(mask, np.where(mask, 0.0, 0.0), "hard", int(mask.sum()), n, spec)

    # Too tight to be a hard mask. Keep the information as a preference so the
    # arm still expresses the LLM's hypothesis, but do not starve the optimiser.
    if int(mask.sum()) > 0:
        return ConstraintOutcome(
            np.ones(n, bool),
            np.where(mask, 0.0, -abs(soft_penalty)),
            "soft",
            int(mask.sum()),
            n,
            spec,
        )

    return ConstraintOutcome(np.ones(n, bool), np.zeros(n), "none", 0, n, spec)


def parse_spec(spec: dict | list) -> Constraint:
    """Build a :class:`Constraint` from the JSON an LLM returns.

    Accepts a list (implicit :class:`AllOf`) or a dict with a ``kind``. Raises
    :class:`ConstraintError` on anything it does not recognise -- silently
    ignoring a clause would let a model that emits nonsense score as if it had
    proposed a sensible constraint.
    """
    if isinstance(spec, list):
        return AllOf(parts=tuple(parse_spec(s) for s in spec))
    if not isinstance(spec, dict):
        raise ConstraintError(f"constraint spec must be a dict or list, got {type(spec).__name__}")

    kind = spec.get("kind")
    if kind in ("substructure", None) and ("require" in spec or "forbid" in spec):
        return SubstructureConstraint(
            require=tuple(spec.get("require", ()) or ()),
            forbid=tuple(spec.get("forbid", ()) or ()),
        )
    if kind == "property" or "bounds" in spec:
        bounds = {}
        for name, rng in (spec.get("bounds") or {}).items():
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise ConstraintError(f"bounds for {name!r} must be [lo, hi], got {rng!r}")
            bounds[name] = (rng[0], rng[1])
        return PropertyConstraint(bounds=bounds)
    if kind == "all_of":
        return AllOf(parts=tuple(parse_spec(p) for p in spec.get("parts", ())))
    if kind == "any_of":
        return AnyOf(parts=tuple(parse_spec(p) for p in spec.get("parts", ())))
    raise ConstraintError(f"unrecognised constraint spec: {spec!r}")

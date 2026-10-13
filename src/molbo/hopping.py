"""Directed scaffold hopping: periodically send BO into an unsampled chemotype.

A Tanimoto GP learns "compounds like the ones I measured are good" and cannot
reason about a scaffold it has never sampled -- its accuracy there is not
merely poor, it is undefined. Measured: at a budget of 30 calls BO touches
3.5-13% of a library's Murcko scaffolds, and 0.7-9.4 of the library's top ten
compounds sit in scaffolds it never enters. Quadrupling the budget does not
fix it; BO deepens the basin it is already in.

So every ``every`` rounds this restricts the candidate pool to one unsampled
scaffold and lets BO pick within it. The budget cost is fixed and small; what
matters is *which* scaffold, and that is the whole finding:

    directed (answer key)  +0.013 to +0.069 AUC, all p < 0.01
    random unsampled       -0.022 to +0.003  -- it LOSES

Exploring more is not the win. Knowing where to explore is. The gap between
those two rows is what a cross-target model has to supply, and unlike a prior
that ranks compounds the GP can already see, it is not competing with a
surrogate that improves -- the GP never learns about regions it does not
sample, so this advantage does not decay with budget. Measured: it grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .constraints import Constraint, LibraryContext


def murcko_labels(smiles: Sequence[str], generic: bool = False) -> np.ndarray:
    """Bemis-Murcko scaffold per molecule, as an integer label.

    ``generic`` ignores atom and bond types, giving the graph framework -- a
    coarser and usually more chemotype-like grouping. Measured on these
    libraries: plain Murcko gives 2.3-3.3 molecules per scaffold with **64-75%
    of scaffolds being singletons**, while generic gives 3.2-5.6.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    keys, out = {}, []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            k = s
        else:
            core = MurckoScaffold.GetScaffoldForMol(m)
            if generic:
                core = MurckoScaffold.MakeScaffoldGeneric(core)
            k = Chem.MolToSmiles(core)
        out.append(keys.setdefault(k, len(keys)))
    return np.array(out)


@dataclass
class HopConstraint(Constraint):
    """Allow only one scaffold (or everything, on a non-hop round)."""

    allowed: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    tag: str = "free"
    kind: str = "scaffold_hop"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        return self.allowed

    def describe(self) -> dict:
        # Must distinguish one hop from another, or LibraryContext.mask_for
        # caches the first scaffold's mask and every hop goes to the same place.
        return {"kind": self.kind, "tag": self.tag}


@dataclass
class ScaffoldHopSource:
    """Send BO into an unsampled scaffold every ``every`` rounds.

    ``score_fn`` ranks scaffold labels; the highest-scoring *unsampled* one is
    entered. Pass ``None`` for a random choice, which is the control that
    matters -- it isolates "knowing where" from "exploring at all".
    """

    labels: np.ndarray
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None
    every: int = 4
    """Rounds between hops. The cost is one round in ``every``, so this is the
    dial between exploitation and directed exploration."""
    min_members: int = 3
    """Only hop into a scaffold with at least this many compounds.

    Not a detail. 64-75% of Murcko scaffolds in these libraries are
    singletons, so an unfiltered random hop usually spends its round on one
    arbitrary molecule and learns nothing about a region -- which is most of
    why the random control loses rather than merely failing to help. A hop
    should buy a foothold in a chemotype, and the member count is known from
    the library alone, with no measurements.
    """

    seen: set = field(default_factory=set, init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        self._scores = (None if self.score_fn is None
                        else np.asarray(self.score_fn(self.labels), dtype=float))
        counts = np.bincount(self.labels, minlength=int(self.labels.max()) + 1)
        self._eligible = {int(c) for c in np.flatnonzero(counts >= self.min_members)}

    def reset(self) -> None:
        self.seen = set()
        self.log = []

    def _free(self) -> HopConstraint:
        return HopConstraint(allowed=np.ones(len(self.labels), bool), tag="free")

    def __call__(self, observations=None, ctx=None, rng=None, round_index: int = 0, **_):
        if ctx is not None:
            pos = {s: i for i, s in enumerate(ctx.smiles)}
            for smi, _v in (observations or []):
                if smi in pos:
                    self.seen.add(int(self.labels[pos[smi]]))
        if round_index % max(1, self.every) or not self.seen:
            return self._free()
        unseen = sorted(self._eligible - self.seen)
        if not unseen:
            return self._free()
        if self._scores is None:
            pick = int(unseen[int((rng or np.random.default_rng(0)).integers(len(unseen)))])
        else:
            pick = int(max(unseen, key=lambda c: self._scores[c]))
        self.log.append({"round": int(round_index), "scaffold": pick,
                         "n_unseen": len(unseen)})
        return HopConstraint(allowed=(self.labels == pick), tag=f"hop{pick}")


def oracle_scores(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Best true value in each scaffold. A ceiling -- it reads the answer key."""
    labels = np.asarray(labels)
    return np.array([values[labels == c].max() if (labels == c).any() else -np.inf
                     for c in range(int(labels.max()) + 1)])


def predicted_scores(labels: np.ndarray, predictions: np.ndarray,
                     agg: str = "max") -> np.ndarray:
    """Score each scaffold by its members' predicted values.

    ``max`` asks "could anything here be good", which is the question a hop
    actually poses -- one excellent compound justifies entering a scaffold
    whose average is poor. ``mean`` asks whether the region is good on the
    whole and is the more conservative read.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions, dtype=float)
    fn = np.max if agg == "max" else np.mean
    return np.array([fn(predictions[labels == c]) if (labels == c).any() else -np.inf
                     for c in range(int(labels.max()) + 1)])

"""DOCKSTRING: 260,148 ligands x 58 targets of docking score, fully dense.

Every other library in this project is one target deep.  That was fine while
the question was *which molecule next*, and it makes the question this module
exists for -- *which molecule, on which target, next* -- unaskable, because a
retrospective benchmark can only reveal ``y[m, t]`` at cells it actually holds.
The repo's own ChEMBL libraries share at most 10% of their compounds pairwise
and 22 of 45 pairs share none; BindingDB is 0.039% dense and its largest
fully-observed block is 1,472 x 4.  DOCKSTRING is 100% dense by construction,
because a docking program returns a number for every pair it is handed.

The cost of that density is that the labels are *predicted* affinities, not
measured ones.  Nothing here should be read as a claim about biology.  What it
buys is a controlled testbed: an arbitrary ``(molecule, target)`` is revealable,
the oracle is noise-free and deterministic, and -- for the LLM arms -- no model
can have memorised an AutoDock Vina score for a specific ZINC ligand the way it
may have memorised that gefitinib inhibits EGFR.

The two objectives are Kubincova & Mobley's, transcribed from
``MultiobjectiveAL.ipynb`` so their numbers are directly comparable:

    f_JAK2 = -JAK2 + min(LCK, -8.1) - 10 (1 - QED)
    f_PPAR = -max(PPARA, PPARD, PPARG) - 10 (1 - QED)

Docking scores are negative-is-better, hence the leading minus: every objective
here is *maximised*, matching the rest of this package.

Download (the plain ``figshare.com/ndownloader`` URL the notebook uses is now
behind an AWS WAF challenge and returns 0 bytes; the API host is not):

    curl -sSL -o data/dockstring/dockstring-dataset.tsv \\
        https://api.figshare.com/v2/file/download/35948138
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

DATASET = Path("data/dockstring/dockstring-dataset.tsv")
"""Default location. 99,384,215 bytes; not committed (see .gitignore)."""

N_LIGANDS = 260_148
"""Rows after ``drop_duplicates(subset="smiles")``, per the notebook."""

TARGETS: tuple[str, ...] = (
    "PPARD", "ABL1", "ADAM17", "ADRB1", "ADRB2", "AKT2", "MAOB", "CASP3",
    "DHFR", "ESR2", "PTK2", "FGFR1", "HMGCR", "HSP90AA1", "KIT", "MAPKAPK2",
    "MAP2K1", "NOS1", "PARP1", "PDE5A", "PGR", "PTPN1", "ROCK1", "AKT1", "AR",
    "CDK2", "CSF1R", "ESR1", "NR3C1", "IGF1R", "JAK2", "LCK", "MET", "MMP13",
    "PTGS2", "PPARA", "PPARG", "REN", "ADORA2A", "ACHE", "BACE1", "CA2",
    "CYP2C9", "CYP3A4", "HSD11B1", "DPP4", "DRD2", "DRD3", "EGFR", "F10",
    "GBA", "MAPK1", "MAPK14", "PLK1", "SRC", "THRB", "F2", "KDR",
)

KINASES: tuple[str, ...] = (
    "ABL1", "AKT1", "AKT2", "CDK2", "CSF1R", "EGFR", "FGFR1", "IGF1R", "JAK2",
    "KDR", "KIT", "LCK", "MAP2K1", "MAPK1", "MAPK14", "MAPKAPK2", "MET",
    "PLK1", "PTK2", "ROCK1", "SRC",
)
"""The kinase subset. A target-allocation experiment needs targets whose
covariance is plausibly non-trivial; an arbitrary 5 of the 58 would mix
proteases, GPCRs and nuclear receptors and make a null uninformative."""


def load(
    path: str | Path = DATASET,
    *,
    with_qed: bool = True,
    n_rows: int | None = None,
):
    """Load the TSV, deduplicate on SMILES, optionally attach QED.

    QED is computed once over 260k molecules (~3 min under RDKit) and cached
    next to the dataset, keyed by the standardiser, because both published
    objectives need it and neither models it -- the notebook treats it as known
    up front, and so does this.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  curl -sSL -o data/dockstring/dockstring-dataset.tsv "
            "https://api.figshare.com/v2/file/download/35948138"
        )
    df = pd.read_csv(path, sep="\t", nrows=n_rows)
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    if with_qed:
        df["QED"] = _qed(df.smiles.tolist(), path, n_rows)
    return df


def _qed(smiles: Sequence[str], path: Path, n_rows: int | None) -> np.ndarray:
    """QED per molecule, memoised to ``<dataset>.qed-<hash>.npy``."""
    key = hashlib.sha1(
        f"{len(smiles)}:{n_rows}:{smiles[0]}:{smiles[-1]}".encode()
    ).hexdigest()[:12]
    cache = path.with_suffix(f".qed-{key}.npy")
    if cache.exists():
        return np.load(cache)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    out = np.full(len(smiles), np.nan)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[i] = QED.qed(mol)
    np.save(cache, out)
    return out


# ---- objectives ----------------------------------------------------------
#
# Transcribed from MultiobjectiveAL.ipynb cells 9 and 13. `var` is anything
# indexable by target name -- a DataFrame or a dict of arrays -- so the same
# function scores the whole library or a single candidate.


QED_WEIGHT = 10.0
"""The published drug-likeness penalty, ``- w (1 - QED)``, at the notebook's w.

At ``w = 10`` this term has 2.0x to 2.7x the spread of the entire docking part
of every published objective, and correlates with the finished objective at rho
+0.87 to +0.93 -- measured in ``scripts/dockstring_audit.py`` check 4, recorded
in FINDINGS.md section 14.  QED is computed from the SMILES, so it costs nothing
and is known for every library member before a single assay: ranking by it alone
recovers ~48% of the top-50 of ``jak2`` with zero assays.  Anything measured on
these objectives is therefore measuring a free descriptor for about half its
value, and ``w`` is the dial that says how much."""

QED_MIN = 0.5
"""Admission threshold for the potency variants. The conventional drug-likeness
cut, used as a *filter* on the library rather than as a term in the score --
which is how a medicinal chemist treats it, and which removes the free component
from the objective instead of merely shrinking it."""


def _jak2_terms(var) -> np.ndarray:
    """JAK2 potency with LCK selectivity saturating past -8.1.

    The ``min(LCK, -8.1)`` is the part that makes this multi-objective rather
    than a weighted sum: once LCK docking is worse than -8.1 the term
    saturates, so there is no credit for being *more* selective than the
    threshold, and the gradient in LCK vanishes. A scalarised surrogate cannot
    see that flat region; a per-target one can.
    """
    return -var["JAK2"] + np.minimum(var["LCK"], -8.1)


def _ppar_terms(var) -> np.ndarray:
    """Promiscuity: bind all three PPAR subtypes at once."""
    return -np.maximum(np.maximum(var["PPARA"], var["PPARD"]), var["PPARG"])


def _jak2_only_terms(var) -> np.ndarray:
    """Single-target control: drop the selectivity term.

    The notebook runs this to separate "multi-target acquisition helps" from
    "this objective is just easier".
    """
    return -var["JAK2"]


@dataclass(frozen=True)
class Objective:
    """A named scalar objective over DOCKSTRING, with the targets it reads.

    The objective is ``assayed(var) - qed_weight * (1 - QED)``, restricted to
    molecules with ``QED >= qed_min``.  Splitting it that way is not cosmetic:
    the first part costs an assay per target per molecule and the second is free
    for the whole library up front, so a benchmark that mixes them reports the
    optimiser's credit for rediscovering a descriptor it was handed.
    """

    name: str
    fn: Callable
    targets: tuple[str, ...]
    """Which docking columns the objective consumes. These are the variables a
    *separated* surrogate would model one at a time, and the axes a
    target-allocation policy would divide a batch across."""

    qed_weight: float = QED_WEIGHT
    qed_min: float | None = None
    """Molecules below this QED are removed from the library by
    :func:`objective_table`, rather than being scored badly. ``None`` admits
    everything, which is what the published objectives do."""

    def assayed(self, var) -> np.ndarray:
        """The part of the score that costs measurements."""
        return np.asarray(self.fn(var), dtype=float)

    def free(self, var) -> np.ndarray:
        """The part known before any assay. Zero when ``qed_weight`` is zero."""
        if self.qed_weight == 0.0:
            return np.zeros_like(np.asarray(var["QED"], dtype=float))
        return -self.qed_weight * (1.0 - np.asarray(var["QED"], dtype=float))

    def admits(self, var) -> np.ndarray:
        """Boolean mask of the molecules this objective is defined on."""
        qed = np.asarray(var["QED"], dtype=float)
        ok = np.isfinite(qed)
        return ok if self.qed_min is None else ok & (qed >= self.qed_min)

    def __call__(self, var) -> np.ndarray:
        return self.assayed(var) + self.free(var)

    def reweighted(self, qed_weight: float, qed_min: float | None = None) -> "Objective":
        """The same assayed terms under a different drug-likeness treatment.

        Used to sweep ``w`` and measure the free-descriptor floor as a function
        of it, so the choice of objective is measured rather than asserted.
        """
        return Objective(f"{self.name}@w{qed_weight:g}", self.fn, self.targets,
                         qed_weight=qed_weight, qed_min=qed_min)


# Published objectives, transcribed as-is. Keep these exactly at w = 10 with no
# filter: they are the comparison to Kubincova & Mobley, not a recommendation.
OBJECTIVES = {
    "jak2": Objective("jak2", _jak2_terms, ("JAK2", "LCK")),
    "ppar": Objective("ppar", _ppar_terms, ("PPARA", "PPARD", "PPARG")),
    "jak2_only": Objective("jak2_only", _jak2_only_terms, ("JAK2",)),
}

# Potency variants: drug-likeness as an admission filter instead of a term, so
# the score is entirely assayed and a free ranking has nothing to stand on.
OBJECTIVES.update({
    f"{k}_potency": Objective(f"{k}_potency", v.fn, v.targets,
                              qed_weight=0.0, qed_min=QED_MIN)
    for k, v in list(OBJECTIVES.items())
})


def score_jak2(var) -> np.ndarray:
    """The published JAK2 objective. See :func:`_jak2_terms`."""
    return OBJECTIVES["jak2"](var)


def score_ppar(var) -> np.ndarray:
    """The published PPAR objective. See :func:`_ppar_terms`."""
    return OBJECTIVES["ppar"](var)


def score_jak2_only(var) -> np.ndarray:
    """The published single-target control. See :func:`_jak2_only_terms`."""
    return OBJECTIVES["jak2_only"](var)


def objective_table(df, objective: str | Objective = "jak2") -> dict[str, float]:
    """``{smiles: score}``, ready for :class:`molbo.oracles.LookupOracle`.

    This is the bridge that makes every existing arm in this package runnable
    on DOCKSTRING without a line of new campaign code: a scalarised
    multi-target objective is, to the optimiser, just another lookup library.

    Molecules the objective does not admit are **absent from the table**, not
    scored low. A filter that leaves rejects in the library at a bad score is
    still an objective term -- the optimiser can spend budget discovering it --
    and that is the thing the potency variants exist to avoid.
    """
    obj = OBJECTIVES[objective] if isinstance(objective, str) else objective
    y = np.asarray(obj(df), dtype=float)
    ok = np.isfinite(y) & obj.admits(df)
    return dict(zip(df.smiles[ok], y[ok]))

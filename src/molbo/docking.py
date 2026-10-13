"""Docking scores as a campaign feature: loader, selector, and the controls.

A Vina score is the only signal in this project that comes from physics rather
than from a model fitted to activity data. Every learned alternative -- the
Tanimoto GP, a ChemBERTa embedding, an ML affinity predictor -- was trained on
the same kind of measurements the benchmark grades against, so a gain from one
of those is always partly a question about memorisation. A force field cannot
have memorised ChEMBL.

What it must overcome is the prior: docking discriminates *worst* inside a
congeneric series, which is exactly what the EGFR anilinoquinazoline library
is. Every compound shares a scaffold and a binding mode, so the pose search has
little to separate them with. Published correlations against measured affinity
are modest on diverse sets and weaker within a series.

So the useful question is not "does docking correlate with potency" but "does
it carry anything the fingerprint GP does not already have". ``incremental``
answers that directly, and :class:`DockSelector` puts it in a campaign where it
can be paired against plain BO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class DockingScores:
    """``smiles -> Vina score`` from ``scripts/dock_library.py``.

    Vina reports kcal/mol where **lower is better**. Everything in this project
    maximises, so :meth:`value` negates. Keeping the raw sign in the file and
    flipping at the boundary means the cache stays readable against Vina's own
    output rather than silently disagreeing with it.
    """

    scores: dict[str, float]
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "DockingScores":
        path = Path(path)
        scores: dict[str, float] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("score") is not None:
                    scores[r["smiles"]] = float(r["score"])
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta and not (meta.get("redock") or {}).get("ok", False):
            raise ValueError(
                f"{meta_path}: the redocking validation did not pass, so these scores are "
                "not trustworthy. Re-run scripts/dock_library.py and check the box."
            )
        return cls(scores=scores, meta=meta)

    def __len__(self) -> int:
        return len(self.scores)

    def value(self, smiles: str, default: float = np.nan) -> float:
        """Higher is better, so a plain ``argmax`` over this means 'best docked'."""
        s = self.scores.get(smiles)
        return default if s is None else -s

    def values(self, smiles: Sequence[str]) -> np.ndarray:
        return np.array([self.value(s) for s in smiles], dtype=float)

    def coverage(self, smiles: Sequence[str]) -> float:
        return float(np.mean([s in self.scores for s in smiles])) if len(smiles) else 0.0


@dataclass
class DockSelector:
    """Buy the shortlist candidate with the best docking score.

    The docking counterpart of :class:`molbo.clm.MaxMeanSelector`: a free,
    fixed rule that uses one extra signal and nothing else. It is the control a
    learnt tool-using selector has to beat, because if simply always taking the
    best-docked candidate works, no policy is needed to decide when to dock.

    Molecules with no cached score fall back to the acquisition ranking rather
    than being dropped, so a partially docked library degrades gracefully into
    plain BO instead of failing.
    """

    docking: DockingScores
    blend_mu: float = 0.0
    """Weight on the GP's posterior mean, in units of its own standard
    deviation. ``0`` is docking alone. Non-zero asks the sharper question --
    whether docking adds anything *on top of* what the surrogate already
    believes, which is the only way it can pay for itself."""

    misses: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)

    def __call__(self, shortlist, mu=None, **_):
        d = self.docking.values(shortlist)
        self.calls += 1
        if np.all(np.isnan(d)):
            self.misses += 1
            return 0
        score = np.where(np.isnan(d), -np.inf, d)
        if self.blend_mu and mu is not None:
            mu = np.asarray(mu, dtype=float)
            z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
            score = np.where(np.isnan(d), -np.inf, z(score) + self.blend_mu * z(mu))
        return int(np.argmax(score))


@dataclass
class LiveDocking:
    """Dock on a cache miss, append the result, carry on.

    The point of docking on demand rather than pre-docking a library is not
    only realism -- nobody can dock chemical space -- it is that the cost then
    **self-limits to what the experiment actually needed**. Arms request
    high-acquisition candidates, which concentrate, and a molecule is paid for
    once across every seed and every arm. Pre-docking 2,103 compounds to
    discover that the arms only ever ask about 300 of them is the expensive way
    to learn the same thing.

    The cache is append-only and shared, so an interrupted run resumes and two
    arms never pay twice for the same answer.
    """

    receptor: str | Path
    box: str | Path
    cache_path: str | Path
    exhaustiveness: int = 8
    seed: int = 42
    cpu: int | None = None
    """``None`` = every core, which is what an on-demand dock wants."""

    scores: dict[str, float] = field(default_factory=dict, init=False)
    meta: dict = field(default_factory=dict, init=False)
    docked: int = field(default=0, init=False)
    failed: int = field(default=0, init=False)

    def __post_init__(self):
        self.cache_path = Path(self.cache_path)
        if self.cache_path.exists():
            loaded = DockingScores.load(self.cache_path)
            self.scores, self.meta = loaded.scores, loaded.meta

    def value(self, smiles: str, default: float = np.nan) -> float:
        """Higher is better. Runs Vina if this molecule has never been docked."""
        if smiles not in self.scores:
            score, note = run_vina(smiles, self.receptor, self.box,
                                   self.exhaustiveness, self.seed, cpu=self.cpu)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"smiles": smiles, "score": score, "note": note}) + "\n")
            if score is None:
                self.failed += 1
                return default
            self.scores[smiles] = score
            self.docked += 1
        return -self.scores[smiles]

    def values(self, smiles: Sequence[str]) -> np.ndarray:
        return np.array([self.value(s) for s in smiles], dtype=float)


def conformer_seed(smiles: str, replicate: int = 0) -> int:
    """A stable RDKit embedding seed for a molecule.

    Python salts ``hash()`` on strings per process (PYTHONHASHSEED), so using
    it here silently produced a different starting conformer on every run --
    and docking is sensitive to that. Measured on erlotinib against its own
    crystal structure: three runs gave poses 0.99 A, 1.08 A and **3.61 A** from
    the crystallographic one. A sha256 digest makes the conformer, and
    therefore the score, reproducible across processes and machines.

    ``replicate`` gives independent conformers for the same molecule, which is
    how the conformer-induced score variance is measured.
    """
    import hashlib

    h = hashlib.sha256(f"{smiles}|{replicate}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def run_vina(smiles: str, receptor, box, exhaustiveness: int = 8, seed: int = 42,
             vina: str | Path | None = None, cpu: int | None = None) -> tuple[float | None, str]:
    """Dock one molecule. -> (Vina score in kcal/mol, note). Lower is better.

    The 3D embedding seed is derived from the SMILES, so the same molecule
    gives the same score on a rerun and a results table is reproducible.

    ``cpu`` is the knob that decides whether this is fast or slow, and the
    right value depends entirely on who is calling. ``None`` lets Vina use
    every core, which is what an on-demand dock inside a campaign wants:
    measured ~5 s per molecule with the machine to itself, against ~108 s
    pinned to one core. A batch job running many molecules in parallel wants
    ``cpu=1`` instead, so the workers do not fight each other.
    """
    import os
    import subprocess
    import tempfile

    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    exe = Path(vina) if vina else Path("tools/vina.exe" if os.name == "nt" else "tools/vina")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable"
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=conformer_seed(smiles)) != 0:
        return None, "no 3D conformer"
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    try:
        txt, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(mol)[0])
    except Exception as exc:
        return None, f"meeko: {type(exc).__name__}"
    if not ok:
        return None, f"meeko: {str(err)[:60]}"
    with tempfile.TemporaryDirectory() as td:
        lig, out = Path(td) / "l.pdbqt", Path(td) / "o.pdbqt"
        lig.write_text(txt)
        cmd = [str(exe), "--receptor", str(receptor), "--ligand", str(lig),
               "--config", str(box), "--out", str(out),
               "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
               "--verbosity", "0"]
        if cpu:
            cmd += ["--cpu", str(cpu)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None, "vina timeout"
        if not out.exists():
            return None, "vina produced no output"
        for line in out.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3]), ""
    return None, "no score in output"


@dataclass
class DockingBudget:
    """Docking as a *scarce* tool: on demand, counted, capped.

    Pre-docking a library makes every score free at decision time, which
    removes the only thing worth studying here -- nobody can dock chemical
    space, and in a real campaign the question is always *which* molecules
    deserve the compute. So the campaign pays per molecule, and what it did
    not pay for it does not know.

    The full cache behind this is the **answer key, not the operating mode**,
    exactly as :class:`~molbo.oracles.lookup.LookupOracle` holds every pIC50
    while a run only sees what it buys. It exists so the ceiling
    ("which molecules *should* have been docked?") is computable at all.

    ``spent`` and ``log`` make the cost visible in the results file: an arm
    that quietly docked ten times more than its control is not a fair
    comparison, and a gap in a table would not show it.
    """

    scores: DockingScores
    budget: int = 50
    """Total molecules this campaign may dock. Compare across arms at equal
    budget, never at equal outcome."""

    spent: int = field(default=0, init=False)
    known: dict[str, float] = field(default_factory=dict, init=False)
    refused: int = field(default=0, init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.spent = 0
        self.known.clear()
        self.refused = 0
        self.log = []

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def request(self, smiles: Sequence[str], round_index: int = 0) -> dict[str, float]:
        """Dock the ones not already known, in order, until the budget runs out.

        Re-requesting a molecule is free -- a score depends only on
        ``(receptor, box, ligand)``, so paying twice for the same answer would
        be an artefact of the harness rather than a real cost.
        """
        bought = []
        for s in smiles:
            if s in self.known:
                continue
            if self.remaining <= 0:
                self.refused += 1
                continue
            v = self.scores.value(s)
            self.known[s] = v
            self.spent += 1
            bought.append(s)
        if bought:
            self.log.append({"round": int(round_index), "n": len(bought),
                             "spent": self.spent, "remaining": self.remaining})
        return self.known

    def values(self, smiles: Sequence[str]) -> np.ndarray:
        """Only what has been paid for; NaN elsewhere."""
        return np.array([self.known.get(s, np.nan) for s in smiles], dtype=float)


@dataclass
class BudgetedDockSelector:
    """Allocate a docking budget across rounds, then pick using what it bought.

    The arm under test for the agentic claim, and the reason the tool set needs
    costs at all. ``allocator`` is the decision:

    ``none``
        Never dock. Reduces to ``MaxMeanSelector`` and pins that the machinery
        itself moves nothing.
    ``top``
        Dock the ``per_round`` highest-acquisition candidates. The free
        heuristic, and the control that matters -- if spending the budget on
        whatever BO already likes works as well as anything, no policy is
        needed.
    ``random``
        Dock ``per_round`` at random from the shortlist. Separates "docking
        helps" from "docking *the right ones* helps".
    ``uncertain``
        Dock where the surrogate is least sure. The textbook answer, and the
        one a learned allocator most plausibly rediscovers.

    A learned allocator replaces this field later; it has to beat ``top`` and
    ``uncertain`` at equal budget, not merely beat ``none``.
    """

    budget: DockingBudget
    per_round: int = 3
    allocator: str = "top"
    blend_mu: float = 1.0
    """Weight on the standardised posterior mean when combining with the
    standardised docking score. Docking measured ~0.03 correlation with the
    GP's mean, so the two are near-orthogonal and blending is the sensible
    default; ``0`` would throw the surrogate away."""

    picks: list[int] = field(default_factory=list, init=False)
    history: list[dict] = field(default_factory=list, init=False)
    """One row per campaign, kept across resets: how much of the allowance
    each seed actually used. An arm that never exhausts its budget is not
    being constrained by it, and the comparison against a cheaper arm would be
    meaningless -- so this has to survive into the results file."""

    def reset(self) -> None:
        if self.budget.spent or self.picks:
            self.history.append({"spent": self.budget.spent,
                                 "refused": self.budget.refused,
                                 "rounds": len(self.budget.log),
                                 "picks": len(self.picks)})
        self.budget.reset()
        self.picks = []

    def _choose(self, shortlist, scores, sd) -> list[str]:
        k = min(self.per_round, len(shortlist), self.budget.remaining)
        if k <= 0 or self.allocator == "none":
            return []
        if self.allocator == "top":
            order = np.argsort(-np.asarray(scores))
        elif self.allocator == "uncertain":
            order = np.argsort(-np.asarray(sd))
        elif self.allocator == "random":
            order = np.random.default_rng(len(self.picks)).permutation(len(shortlist))
        else:
            raise ValueError(f"unknown allocator {self.allocator!r}")
        return [shortlist[i] for i in order[:k]]

    def __call__(self, shortlist, scores=None, mu=None, sd=None, round_index: int = 0, **_):
        shortlist = list(shortlist)
        if scores is None:
            scores = np.zeros(len(shortlist))
        self.budget.request(self._choose(shortlist, scores, sd), round_index)
        d = self.budget.values(shortlist)
        mu = np.asarray(mu, dtype=float) if mu is not None else np.zeros(len(shortlist))

        def z(v):
            v = np.asarray(v, dtype=float)
            s = np.nanstd(v)
            return (v - np.nanmean(v)) / (s + 1e-9)

        combined = self.blend_mu * z(mu)
        if np.isfinite(d).any():
            # Undocked candidates keep their surrogate score alone rather than
            # being penalised: not having paid to look is not evidence against
            # a molecule.
            combined = combined + np.nan_to_num(z(d), nan=0.0)
        pick = int(np.argmax(combined))
        self.picks.append(pick)
        return pick


def incremental(docking: np.ndarray, mu: np.ndarray, true: np.ndarray) -> dict:
    """Does docking explain anything the surrogate's mean does not?

    Ordinary least squares of ``true`` on ``mu`` alone, then on ``mu`` plus the
    docking score, reported as the change in R-squared and the partial
    (Spearman) correlation of the residuals. A docking score that merely agrees
    with the GP adds nothing here, however well it correlates on its own -- and
    agreeing with the GP is the likely case, since both reward the same bulk
    similarity to known actives.
    """
    from scipy.stats import spearmanr

    ok = np.isfinite(docking) & np.isfinite(mu) & np.isfinite(true)
    d, m, y = docking[ok], mu[ok], true[ok]
    if len(y) < 10:
        return {"n": int(len(y)), "error": "too few complete rows"}

    def r2(X):
        X = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return 1.0 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum()), resid

    r2_mu, resid_mu = r2(m)
    r2_both, _ = r2(np.column_stack([m, d]))
    return {
        "n": int(len(y)),
        "rho_dock_true": float(spearmanr(d, y).statistic),
        "rho_dock_mu": float(spearmanr(d, m).statistic),
        "r2_mu": r2_mu,
        "r2_mu_plus_dock": r2_both,
        "delta_r2": r2_both - r2_mu,
        "partial_rho": float(spearmanr(d, resid_mu).statistic),
    }

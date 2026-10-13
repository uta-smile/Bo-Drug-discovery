"""Does the transfer gain survive as a *campaign* result? The contribution test.

    python scripts/dense_block_campaign.py --out results/dense_block_campaign.json

Sections 18-20 measure one thing: Spearman of a surrogate's posterior against
truth. That is not the claim this project makes. The claim is about Bayesian
optimisation -- finding potent compounds under a fixed assay budget -- and a
surrogate that ranks the whole library better does not automatically find the
top of it faster. Rank correlation is dominated by the bulk; a campaign lives or
dies on the tail, and the two come apart often enough that the repo already has
a section about offline proxies inverting against campaign outcome (section 4).

So this runs the actual loop. Same dense block, same free-history setup as
section 18's check, but the surrogate now drives acquisition:

* **stl** -- Tanimoto GP on the primary target alone. Standard BO.
* **mt**  -- coregionalised GP with ``H`` free historical observations per
  auxiliary target, fitted `r`. The arm sections 18-19 support.
* **pooled** -- same observations, ``r`` fixed at 1. Kept because it is the only
  control that separates "modelling the task structure helped" from "more rows
  helped"; on JAK it is *worse* than using no auxiliary data at all, so it is
  not a formality.

Every arm gets the identical initial design, budget, batch size and acquisition;
only the surrogate differs. Reported as top-k recovery and as AUC of the top-k
trace, the metrics the rest of this package uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from molbo.acquisition import ACQUISITIONS, BETA, EI_XI, acquisition_scores  # noqa: E402

ACQ_PORTFOLIO = ACQUISITIONS
"""The action space the `oracle` ceiling chooses from, per round.

Measured inside real multi-task campaigns, these five are not five
distinct actions. Against EI's own batch, UCB and PI agree at Spearman
0.97 and 0.996 with 67-79% of the same top-10 picks, `greedy` at 0.74-0.80
with 28-38%, and `ts` at 0.39-0.45 with 8-15%. So the portfolio is really
about two and a half choices, and a near-zero ceiling over it is a fact
about the action space rather than about adaptivity.
"""
from molbo.audit import Audit  # noqa: E402
from molbo.metrics import auc_top_k, degrade_to_rho  # noqa: E402
from molbo.surrogate import CoregionalisedGP, TanimotoGP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_block_h1 import (  # noqa: E402
    SOURCE,
    build_block,
    compute_qed,
    drop_censored,
    drop_non_druglike,
    nanomolar_potency,
    pareto_mask,
)

SIM_ACTIVES = 10
"""How many known actives the similarity baseline ranks against. Ten matches
the top-k being scored, so the heuristic is given the same notion of "good"
the metric uses."""


def _z(v):
    """z-score, so an auxiliary target on a different scale is comparable.

    The similarity baseline pools actives across targets; without this a target
    whose pIC50 happens to run higher would supply every "known active".
    """
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


FAMILIES = {
    "jak": {"P23458": "JAK1", "O60674": "JAK2", "P52333": "JAK3", "P29597": "TYK2"},
    "cyp": {"P08684": "CYP3A4", "P10635": "CYP2D6", "P11712": "CYP2C9",
            "P33261": "CYP2C19"},
    # The eight below were not picked by hand. `scripts/find_dense_blocks.py`
    # scans the whole BindingDB IC50 dump for 4-target sets with the largest
    # fully-observed sub-matrix; jak and cyp come out first and second, which
    # is the check that the scan agrees with what sections 18-25 already used,
    # and these are the next eight that clear 300 compounds and are not a
    # near-duplicate of one already listed. Keeping them means the operating
    # window (section 25) is stated over ten protein families spanning six
    # mechanistic classes rather than two.
    "pi3k": {"P42336": "PIK3CA", "P42338": "PIK3CB", "O00329": "PIK3CD",
             "P48736": "PIK3CG"},
    "dpp": {"P27487": "DPP4", "Q9UHL4": "DPP7", "Q6V1X1": "DPP8",
            "Q86TI2": "DPP9"},
    "abc": {"O95342": "ABCB11", "Q92887": "ABCC2", "O15438": "ABCC3",
            "O15439": "ABCC4"},
    "hdac": {"Q13547": "HDAC1", "Q92769": "HDAC2", "O15379": "HDAC3",
             "Q9UBN7": "HDAC6"},
    "fgfr": {"P11362": "FGFR1", "P21802": "FGFR2", "P22607": "FGFR3",
             "P22455": "FGFR4"},
    "mmp": {"P03956": "MMP1", "P08253": "MMP2", "P08254": "MMP3",
            "P14780": "MMP9"},
    "sstr": {"P30872": "SSTR1", "P30874": "SSTR2", "P31391": "SSTR4",
             "P35346": "SSTR5"},
    "cathepsin": {"P07858": "CTSB", "P43235": "CTSK", "P07711": "CTSL",
                  "P25774": "CTSS"},
}
"""Dense co-measured blocks: UniProt -> label, one entry per protein family.

Mechanistic class per family, because the claim these support is that the
effect is not specific to one kind of protein: jak and fgfr are receptor/
non-receptor protein kinases, pi3k a lipid kinase, cyp a heme monooxygenase
panel, hdac a zinc deacetylase, mmp a zinc protease, dpp a serine protease,
cathepsin a cysteine protease, sstr a GPCR, abc an ATP-binding transporter.
"""


PANELS = {
    "kiba": {"source": Path("data/cache/tdc/kiba_score.csv"),
             "potency": "identity"},
    # Kd in nM: LOWER is better, so this needs the nanomolar transform, not
    # identity. KIBA is the opposite case -- its score is already an integrated
    # higher-is-better statistic -- and the two sitting side by side is exactly
    # how `identity` got copied onto this row. With `identity` a campaign would
    # have maximised Kd, i.e. optimised for the weakest binders, and 71% of
    # Davis sits at the 10,000 nM "no binding" ceiling, so the top-k would have
    # been the non-binders. Nothing stored used it; the registry is reachable
    # only through --blocks-from and no Davis block was ever pinned.
    "davis": {"source": Path("data/cache/tdc/davis_kd.csv"),
              "potency": "nanomolar"},
}
"""Kinome profiling panels from ``fetch_kinase_panels.py``, with the ``--potency``
each one needs pinned to it.

The pairing is the whole point. ``FAMILIES`` is safe to name on its own because
every entry is cut from the BindingDB IC50 dump, where ``nanomolar`` is always
right. A panel's ``Y`` is not an IC50, and the two settings are not symmetric in
their failure: pointing ``nanomolar`` at a KIBA score silently *inverts* the
objective and optimises for the weakest binders, which no audit check catches
because every invariant still holds. Writing the pair down once is cheaper than
getting it right at each call site.
"""

PANEL_FAMILIES = {
    "kiba_prkc": {"O94806": "PRKD3", "Q05513": "PRKCZ", "Q05655": "PRKCD",
                  "P05129": "PRKCG"},
    "kiba_cdk5": {"P11309": "PIM1", "P17612": "PRKACA", "Q00535": "CDK5",
                  "Q13627": "DYRK1A"},
    "kiba_lck": {"P06239": "LCK", "P45983": "MAPK8", "P45984": "MAPK9",
                 "P49841": "GSK3B"},
}
"""The three KIBA blocks a campaign has actually been run on, pinned by UniProt.

They were first run through ``--blocks-from results/dense_blocks_kiba.json`` as
``k01``, ``k02`` and ``k10``, and those labels are **positions in a scan**, not
names: re-run `find_dense_blocks` at a different ``--min-per-target`` and ``k10``
is a different set of proteins, which would make the stored results point at the
wrong block with nothing to show that they had moved. The accession sets here are
the same ones, read back from that scan artefact and checked against the target
lists in ``results/kiba_block_rho.json``. ``--blocks-from`` still works and is
still the way to reach the other twelve; this is for the three that have results
attached to them.

They span the correlation range the panel was fetched to probe -- within one
paralogue group (PRKC, median rho 0.99), across adjacent groups (0.78), and
across families (LCK/MAPK/GSK3B, 0.48).

The name's prefix is the ``PANELS`` key, which is how ``--source`` and
``--potency`` get filled in; a new entry has to be named ``<panel>_<label>`` or
it will not resolve.
"""


def block_provenance(args, *, targets, block_csv, n_block, n_pool):
    """What a stored result needs in order to name the data it came from.

    A report carrying ``"family": "k10"`` and nothing else is not reproducible:
    the label resolves through ``--blocks-from``, the values depend on
    ``--potency``, and whether the censoring filter ran at all is the difference
    between the CYP block's 335 usable compounds and its 1,460 raw ones. None of
    those were written down, so a KIBA run and a BindingDB run are
    indistinguishable in ``results/`` once the terminal scrollback is gone. This
    is the same defect twice over: section 30 found the window table had reached
    a paper unaudited because `--audit` printed its verdict and stored nothing.

    Additive by construction -- a new key on the report, no existing key touched,
    so every stored JSON still parses and every number in them still stands.
    ``getattr`` throughout, so the sibling scripts can call it with their own
    smaller argument sets.
    """
    # as_posix, not str: these are compared between runs and quoted in the
    # write-up, and a backslash copy would not match one made on Linux.
    return {
        "source": Path(getattr(args, "source", "")).as_posix(),
        "potency": getattr(args, "potency", "nanomolar"),
        "blocks_from": getattr(args, "blocks_from", None),
        "seed_pool": getattr(args, "seed_pool", "random"),
        "targets": list(targets),
        "block_csv": Path(block_csv).as_posix(),
        "n_block": int(n_block),
        "n_pool": int(n_pool),
        "censoring": ("skipped (--keep-censored)"
                      if getattr(args, "keep_censored", False)
                      else f"pile_frac={getattr(args, 'pile_frac', None)}"),
        "qed_min": getattr(args, "qed_min", None),
        "subsample": getattr(args, "subsample", None),
    }


class RunDigest:
    """Content hashes of what a run *fed* its arms, for cross-run comparison.

    Two runs that differ only in a null (``--aux-shuffle`` against
    ``--aux-quality``) are a matched pair only if they assayed the same
    history rows from the same starting designs. Within one run the ``Audit``
    already enforces that every arm sees one shared history; nothing checked it
    *between* runs, and the statistics that compare them are paired. Equal
    coverage statistics -- ``reach``, the history size -- are much weaker than
    equality: two different draws can report the same reach.

    So hash the thing itself. Three streams, each fed in cell order:

    ``history``      the drawn indices, per auxiliary slot, in draw order.
    ``aux_value``    the values handed to the arms at those indices -- the one
                     stream a null is *supposed* to change.
    ``init_design``  the initial design, taken from the first arm (the audit
                     holds the others identical to it).

    A matched pair must agree on ``history`` and ``init_design`` and differ on
    ``aux_value``; identical ``aux_value`` digests across a null would mean the
    null never took effect. Int64 and float64 are written little-endian
    explicitly so a digest means the same thing on another machine.
    """

    STREAMS = ("history", "aux_value", "init_design")

    def __init__(self):
        self._h = {s: hashlib.sha256() for s in self.STREAMS}
        self._by_target = {}
        self.cells = 0

    def _feed(self, stream, target, key, arrays, dtype):
        for h in (self._h[stream],
                  self._by_target.setdefault((stream, target),
                                             hashlib.sha256())):
            h.update(f"{key}|".encode())
            for a in arrays:
                h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
                h.update(b";")

    def cell(self, target, ti, r, *, aux_idx, aux_val):
        """One (target, repeat) cell's history, before any arm has run."""
        self.cells += 1
        key = f"{ti}:{r}"
        self._feed("history", target, key, aux_idx, "<i8")
        self._feed("aux_value", target, key, aux_val, "<f8")

    def init_design(self, target, ti, r, picked):
        self._feed("init_design", target, f"{ti}:{r}", [picked], "<i8")

    def to_json(self):
        out = {s: self._h[s].hexdigest() for s in self.STREAMS}
        out["cells"] = self.cells
        out["by_target"] = {
            s: {t: h.hexdigest()
                for (s_, t), h in self._by_target.items() if s_ == s}
            for s in self.STREAMS}
        return out


def draw_history(K, y, size, rng, *, cluster_frac, exclude_top, k):
    """Historical compounds, at a controllable distance from the search pool.

    Section 22 bounded the campaign gain between two endpoints -- history drawn
    uniformly from the library being searched, and history forbidden from
    containing the answer -- without saying where a real case falls. Overlap is
    measurable, though: ``cold_target_chembl.py`` puts real per-target ChEMBL
    libraries at a reach ratio of 0.53 and a co-measured block at 1.0.

    So make it a knob. ``cluster_frac`` of the history is taken from a tight
    neighbourhood of a random anchor and the rest uniformly; sweeping it walks
    the history from "a different chemical series" to "a random sample of the
    same library". The achieved reach is measured per draw rather than assumed,
    because the knob is not the quantity -- the ratio is.
    """
    n_pool = len(y)
    pool_idx = np.arange(n_pool)
    if exclude_top:
        pool_idx = np.setdiff1d(pool_idx, np.argsort(y)[-k:])
    n_cluster = int(round(cluster_frac * size))
    if n_cluster:
        anchor = int(rng.choice(pool_idx))
        order = pool_idx[np.argsort(-K[anchor, pool_idx])]
        clustered = order[:n_cluster]
    else:
        clustered = np.empty(0, dtype=np.intp)
    rest = np.setdiff1d(pool_idx, clustered)
    unif = rng.choice(rest, size=size - n_cluster, replace=False)
    return np.concatenate([clustered, unif]).astype(np.intp)


def reach_ratio(K, hist):
    """Mean nearest-neighbour similarity to the history, over the pool's own.

    1.0 means the history sits in the same chemistry the campaign is searching;
    the value drops as it moves into a series of its own. Comparable to the
    numbers in ``cold_target_chembl.py``.
    """
    n = K.shape[0]
    cand = np.setdiff1d(np.arange(n), hist)
    own = K[np.ix_(cand, cand)].copy()
    np.fill_diagonal(own, -1.0)
    denom = float(own.max(axis=1).mean())
    if denom <= 0:
        return float("nan")
    return float(K[np.ix_(cand, hist)].max(axis=1).mean() / denom)


def parego_scores(mu, qed_cand, obs_y, rng, *, rho=0.05):
    """Augmented Tchebycheff scalarisation with a fresh random weight.

    Two objectives, maximised: predicted potency and QED. ParEGO's device is to
    redraw the weight every round, so the sequence of picks sweeps the front
    rather than converging on one corner of it. The weight is redrawn, not
    fitted: there is no preference between potency and drug-likeness to fit.

    Potency is normalised by the range *observed so far*, never by the library's
    true range, which would leak the answer. QED is normalised by the library
    range because it genuinely is known for every compound in advance --- that
    asymmetry is the whole point of treating it as free.
    """
    w = rng.random()
    w = np.array([w, 1.0 - w])
    lo, hi = float(obs_y.min()), float(obs_y.max())
    f_pot = (mu - lo) / (hi - lo) if hi > lo else np.zeros_like(mu)
    F = np.column_stack([np.clip(f_pot, -1.0, 2.0), qed_cand])
    g = 1.0 - F                      # to minimisation, both in [0, 1]
    wg = g * w[None, :]
    return -(wg.max(axis=1) + rho * wg.sum(axis=1))


def scaffold_disjoint_pool(block, y, k, rng, scaf=None):
    """Rows whose Bemis-Murcko scaffold appears nowhere in the true top-k.

    A random initial design can seed a campaign with a near-analogue of the
    answer, and on a library assembled as congeneric series that is most of the
    search. Restricting the seed to scaffolds absent from the top-k forces the
    campaign to cross scaffolds to succeed, which is the retrospective analogue
    of screening a catalogue you have no series in.

    This is the admission rule the virtual-screening literature calls a scaffold
    split, applied to the *initial design* rather than to a train/test split,
    because a campaign has no fixed training set -- it chooses one.
    """
    scaf = murcko_labels(block) if scaf is None else scaf
    top = np.argsort(y)[-k:]
    banned = {scaf[int(i)] for i in top}
    ok = np.array([sc not in banned for sc in scaf])
    return np.flatnonzero(ok), len(banned), len(set(scaf))


def murcko_labels(block):
    """Bemis-Murcko scaffold per row, computed once per block.

    Both seeding rules below need it and so does ``scaffold_disjoint_pool``,
    and it is the expensive part of all three -- RDKit parses every SMILES.
    Compounds RDKit cannot give a scaffold get a unique placeholder rather than
    a shared one, so they never match each other by accident.
    """
    from molbo.chem import murcko_scaffold

    return [murcko_scaffold(s) or f"__none_{i}"
            for i, s in enumerate(block.smiles.tolist())]


def history_seed_pool(K, scaf, aux_idx, aux_val, *, mode, n_top, n_nn):
    """Start the campaign near what the relatives already say is good.

    The free history is a set of measured cells on auxiliary targets. A plain
    campaign ignores it when choosing where to begin and draws the initial
    design uniformly; the multi-task surrogate uses it, but only *after* the
    first batch has been bought. These two rules spend it earlier, at the one
    moment the surrogate has nothing:

    ``aux_top``
        the ``n_top`` compounds each auxiliary scored highest on, unioned.
        Per-target z-scores, because a relative whose values happen to run
        higher would otherwise supply the whole pool -- the same reason ``_z``
        exists for the similarity baseline.
    ``aux_scaffold_nn``
        those, plus every compound sharing a Bemis-Murcko scaffold with one,
        plus each one's ``n_nn`` nearest Tanimoto neighbours. Deliberately a
        superset of ``aux_top``: the pair answers whether *widening* a good seed
        region helps or whether the winners themselves were the whole of it.

    No primary value is read anywhere here, and the only auxiliary numbers used
    are the ones the history actually exposes -- so a run with
    ``--aux-quality`` seeds off the degraded values, which is what an
    experimenter holding a computed auxiliary would really have.

    **What this cannot check.** If the history was not drawn clear of the
    primary's top-k, a seeding rule can land directly on the answer. That is
    legitimate in deployment -- knowing a compound's JAK1 number is real prior
    information -- but it is not a chemistry result, so the caller is handed
    the overlap and should report it next to the gain.
    """
    winners: list[int] = []
    for idx, val in zip(aux_idx, aux_val):
        v = _z(val)
        order = np.argsort(v)[-min(n_top, len(v)):]
        winners.extend(int(idx[int(o)]) for o in order)
    winners = sorted(set(winners))
    if mode == "aux_top":
        pool = winners
    elif mode == "aux_scaffold_nn":
        hot = {scaf[i] for i in winners}
        pool = {i for i, sc in enumerate(scaf) if sc in hot}
        pool.update(winners)
        for w in winners:
            # argsort ascending, so the last n_nn+1 are the most similar and
            # the final one is the compound itself.
            nn = np.argsort(K[w])[-(n_nn + 1):]
            pool.update(int(j) for j in nn)
        pool = sorted(pool)
    else:
        raise ValueError(f"unknown seeding mode {mode!r}")
    return np.asarray(pool, dtype=np.intp), {"n_winners": len(winners)}


def audit_campaign(trace, *, arm, budget, k, y, aux_idx, eligible, excl_top,
                   audit):
    """Check at runtime the invariants the comparison's fairness rests on.

    Each of these is asserted somewhere in the write-up, and none of them was
    being checked where the code runs. They are cheap, and a violation of any
    one would silently invalidate the headline numbers rather than crash.

    The checks themselves live in :mod:`molbo.audit`, so this script and the
    regression instruments run the same code and report in the same format.
    """
    idx = np.asarray(trace, dtype=np.intp)
    audit.exactly(idx, budget, f"{arm} selections")
    if eligible is not None:
        audit.subset(idx, eligible, f"{arm} picks", "the eligible set")
    if excl_top:
        # The history is meant to have been drawn clear of the answer; if it
        # was not, the gain is partly the history pointing at the top-k.
        top = np.argsort(y)[-k:]
        hist = np.concatenate(aux_idx) if aux_idx else np.empty(0, dtype=np.intp)
        audit.disjoint(top, hist, f"{arm} primary top-k", "the free history")


def shuffle_auxiliary_responses(vals, aux_idx, others, rng, scope="library"):
    """Return permuted histories and explicit source-index provenance.

    Library scope reproduces the original full-column permutation. History
    scope permutes only the responses actually observed at each auxiliary task.
    Neither mode changes the historical locations or the primary responses.
    """
    if scope not in ("library", "history"):
        raise ValueError("permutation scope must be library or history")
    responses, records = [], []
    for idx, target in zip(aux_idx, others):
        idx = np.asarray(idx, dtype=np.intp)
        if scope == "library":
            order = rng.permutation(len(vals))
            full = vals[order, target]
            response = full[idx]
            source_idx = order[idx]
            correlation = float(spearmanr(full, vals[:, target]).statistic)
        else:
            order = rng.permutation(len(idx))
            source_idx = idx[order]
            response = vals[source_idx, target]
            original = vals[idx, target]
            assert np.array_equal(np.sort(original), np.sort(response))
            correlation = float(spearmanr(response, original).statistic)
        responses.append(response)
        records.append({"scope": scope, "target_column": int(target),
                        "locations": idx.tolist(), "source_indices": source_idx.tolist(),
                        "achieved_correlation": correlation})
    return responses, records


def run_campaign(K, y, aux_idx, aux_val, *, arm, budget, batch, n_init, k,
                 acquisition, rng, qed=None, eligible=None, seed_pool=None,
                 init_idx=None, sd_scale=1.0, avoid_weight=None, trace=None):
    """One campaign. Returns the sequence of true values, in acquisition order.

    The auxiliary observations are fixed before the campaign starts and never
    grow: they are history, not budget. Only the primary target's picks are
    charged, so every arm spends exactly ``budget`` assays on the target it is
    being scored on.

    ``init_idx`` hands the initial design in rather than drawing one, and is
    what lets `init_headroom.py` score several initial designs against each
    other on an otherwise identical run: the rng is then untouched by the draw,
    so the tie-breaking jitter downstream is the same stream for every
    candidate and the only difference between them is where they started.
    Default ``None`` draws as before, byte for byte.

    ``sd_scale`` multiplies the posterior standard deviation before the
    acquisition reads it, leaving the posterior mean alone. It is a control, not
    a method: section 32 found a pure-noise auxiliary buying recall under EI and
    blamed uncertainty compression, and `null_mechanism.py` measured the
    compression at a rung where the task covariance is diagonal. Setting this to
    the measured ratio turns `stl` into the arm that compression alone would
    produce, so a reader can see whether the noise arm's gain is reachable
    without any auxiliary data at all. At 1.0 the multiply is skipped rather
    than applied, so stored runs reproduce exactly.

    ``trace`` is an optional callback invoked once per selection round with the
    quantities needed to say *why* a compound was chosen: the eligible set, the
    primary-observed set, the posterior mean and standard deviation and the
    acquisition value over the eligible set, and what was selected. It exists
    because two different mechanisms produce the same outward behaviour -- a
    model can avoid the auxiliary history because its acquisition value there
    is suppressed, or the history can simply have been removed from the
    candidate set -- and only the second is a bug. For the record: nothing here
    removes the history. ``cand`` is every compound not yet *assayed on the
    primary target*, so a compound measured only on an auxiliary target stays
    eligible, which is the problem setting this paper intends. Default ``None``
    skips the callback, so stored runs are unaffected.
    """
    n_pool = len(y)
    # Union of the compounds some auxiliary target was measured on. Identities
    # only -- no values are read here, which is what makes `stl_avoid` a
    # selection control rather than a transfer method.
    hist_idx = (np.unique(np.concatenate([np.asarray(a, dtype=np.intp)
                                          for a in aux_idx]))
                if aux_idx else np.empty(0, dtype=np.intp))
    start = np.arange(n_pool) if eligible is None else np.asarray(eligible)
    if seed_pool is not None:
        # Only the *initial design* is restricted. The campaign may then select
        # anything, which is the point: it has to reach the answer's chemistry
        # rather than start inside it.
        start = np.intersect1d(start, seed_pool)
    observed = (list(np.asarray(init_idx, dtype=np.intp)[:n_init])
                if init_idx is not None
                else list(rng.choice(start, size=n_init, replace=False)))
    while len(observed) < budget:
        obs = np.asarray(observed, dtype=np.intp)
        cand = np.setdiff1d(np.arange(n_pool), obs)
        if eligible is not None:
            # Spend the budget only on compounds that can count as successes,
            # while the surrogate still sees free historical data from the
            # whole library. Cutting the library up front throws that history
            # away for no saving, because history is not charged.
            cand = np.intersect1d(cand, eligible)
        if arm == "random":
            # The floor. Without it, "better than single-task BO" does not say
            # whether any of the modelling earned its place.
            take = min(batch, budget - len(observed))
            picked = rng.choice(cand, size=take, replace=False)
            if trace is not None:
                # Traced too: "random from the eligible set" is the condition
                # that says whether a design advantage needs a model at all.
                trace({"round": (len(observed) - n_init) // max(batch, 1),
                       "n_observed": len(observed), "eligible": cand,
                       "observed": obs, "mu": None, "sd": None, "acq": None,
                       "selected": picked})
            observed.extend(picked.tolist())
            continue
        if arm == "qed_only":
            # The free arm: rank by drug-likeness alone, spend no information
            # on potency. It should find the high-QED end of the Pareto front
            # and miss the potent end entirely. If it does not, the
            # multi-objective benchmark has the section 14 disease.
            take = min(batch, budget - len(observed))
            order = np.argsort(-(qed[cand] + 1e-12 * rng.standard_normal(len(cand))))
            observed.extend(cand[order[:take]].tolist())
            continue
        if arm == "similarity":
            # What a medicinal chemist does without a model: rank the library by
            # similarity to the actives you already know about, test the top.
            # It uses exactly the information the multi-task arm uses -- the
            # history and the measurements so far -- and costs nothing, so it is
            # the baseline the contribution actually has to clear.
            known = [(float(v), int(i)) for idx, val in zip(aux_idx, aux_val)
                     for i, v in zip(idx, _z(val))]
            known += [(float(v), int(i)) for i, v in zip(obs, _z(y[obs]))]
            top = [i for _, i in sorted(known, reverse=True)[:SIM_ACTIVES]]
            scores = K[np.ix_(cand, np.asarray(top, dtype=np.intp))].max(axis=1)
            take = min(batch, budget - len(observed))
            order = np.argsort(-(scores + 1e-12 * rng.standard_normal(len(scores))))
            observed.extend(cand[order[:take]].tolist())
            continue
        if arm in ("stl", "stl_avoid"):
            gp = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], y[obs])
            mu, sd = gp.predict_gram(K[np.ix_(cand, obs)],
                                     k_diag=np.ones(len(cand)), return_std=True)
        else:
            # `mt_r0` is the separation the mechanism question needs. It is the
            # joint model with the task covariance forced to be diagonal, so
            # the auxiliary rows contribute no cross-target covariance at all,
            # while the shared hyperparameters are still fitted on the joint
            # data. Any gain it shows cannot be transfer; it is what refitting
            # on more rows does on its own.
            grid = {"pooled": (1.0,), "mt_r0": (0.0,)}.get(arm)
            model = (CoregionalisedGP(rho_grid=grid) if grid is not None
                     else CoregionalisedGP())
            model.fit_gram(K, [obs] + aux_idx, [y[obs]] + aux_val)
            mu, sd = model.predict_gram(K, cand, return_std=True)
        if sd_scale != 1.0:
            # The whole control, in one line. Only the acquisition sees this;
            # the posterior mean, the fit and the scoring are untouched.
            sd = sd * sd_scale
        if qed is not None:
            # Multi-objective: the surrogate supplies predicted potency, QED is
            # known exactly, and the scalarisation is re-randomised per round.
            # Only the potency predictor differs between stl and mt.
            scores = parego_scores(mu, qed[cand], y[obs], rng)
        elif acquisition in ("switch_random", "switch_rr"):
            # The free controls. A switching schedule that uses no information
            # at all: uniform at random, or straight round-robin. FINDINGS 1
            # runs these alongside the ceiling for one reason -- an LLM number
            # means nothing until it is read against what costs nothing, and on
            # the single-task library both of these *lost* to the best fixed
            # arm (-0.0057 and -0.0045 at budget 50). If a control matches the
            # ceiling here, the ceiling is reachable without a controller and
            # there is nothing for one to earn.
            rounds = (len(observed) - n_init) // max(batch, 1)
            kind = (ACQ_PORTFOLIO[rounds % len(ACQ_PORTFOLIO)]
                    if acquisition == "switch_rr"
                    else str(rng.choice(ACQ_PORTFOLIO)))
            scores = acquisition_scores(kind, mu, sd,
                                        best=float(y[obs].max()), rng=rng)
        elif acquisition == "oracle":
            # CEILING, NOT A METHOD. It reads `y`, which no arm may do, and is
            # labelled here exactly as `OraclePolicy` and `oracle_constraint`
            # are: a reference line saying how much room an acquisition chooser
            # has, so a small LLM effect can be read against the size of the
            # prize.
            #
            # Greedy over rounds: it picks the acquisition whose own batch has
            # the best mean true value *this* round. That makes it a LOOSE
            # bound -- a controller with foresight could take a worse batch now
            # for a better posterior later, and section 1's artefact shows this
            # construction landing BELOW the best fixed arm at budget 100
            # (-0.0158). So read it only in one direction: if even this gains
            # nothing the action space is inert, but if it loses that is an
            # artefact of myopia and not evidence against adaptivity.
            take_now = min(batch, budget - len(observed))
            best_val, scores = -np.inf, None
            for kind in ACQ_PORTFOLIO:
                # A fresh generator per kind: `ts` consumes randomness, and
                # scoring one kind must not shift what another would have seen.
                s = acquisition_scores(kind, mu, sd, best=float(y[obs].max()),
                                       rng=np.random.default_rng(len(observed)))
                pick = cand[np.argsort(-s)[:take_now]]
                v = float(np.mean(y[pick]))
                if v > best_val:
                    best_val, scores = v, s
        else:
            scores = acquisition_scores(acquisition, mu, sd,
                                        best=float(y[obs].max()), rng=rng)
        if arm == "stl_avoid":
            # History avoidance with no auxiliary *values* anywhere in the run.
            # `stl` above fitted on the primary observations alone, so the only
            # thing this arm adds over it is the identity of the compounds some
            # relative was measured on -- demoted in the ranking, never made
            # ineligible, so the budget and the eligible set match every other
            # arm exactly.
            #
            # It exists to answer one question the pooled arm cannot: the
            # pooled model both borrows auxiliary values and shies away from
            # auxiliary-measured compounds, and those two are confounded in its
            # recall. This arm has the second without the first, so whatever it
            # reproduces of the withheld-vs-uniform reversal is attributable to
            # selection, not to transfer.
            #
            # `--avoid-weight` makes the strength tunable: EI is non-negative,
            # so multiplying by W in [0, 1] interpolates from `stl` (1.0) to
            # near-total avoidance (0.0). Default is the strict endpoint, -inf,
            # which also outranks a tie at zero EI. Only this arm is affected,
            # so the `stl` baseline in the same run stays a true reference.
            mask = np.isin(cand, hist_idx)
            scores = np.where(mask,
                              -np.inf if avoid_weight is None
                              else scores * avoid_weight,
                              scores)
        take = min(batch, budget - len(observed))
        # Ties broken by a fixed jitter draw rather than argsort order, which
        # would otherwise favour low indices and quietly correlate every arm.
        order = np.argsort(-(scores + 1e-12 * rng.standard_normal(len(scores))))
        picked = cand[order[:take]]
        if trace is not None:
            trace({"round": (len(observed) - n_init) // max(batch, 1),
                   "n_observed": len(observed), "eligible": cand,
                   "observed": obs, "mu": mu, "sd": sd, "acq": scores,
                   "selected": picked})
        observed.extend(picked.tolist())
    idx = np.asarray(observed, dtype=np.intp)
    return y[idx], idx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Validated against the resolved registry below rather than with `choices`,
    # because --blocks-from swaps that registry for a scan artefact's blocks and
    # argparse fixes `choices` at parser-construction time.
    p.add_argument("--family", default="jak",
                   help="a registry family, or a block label from --blocks-from")
    p.add_argument("--blocks-from", metavar="SCAN.json",
                   help="take blocks from a find_dense_blocks scan (labels k01, "
                        "k02, ...) instead of the hand-labelled registry")
    p.add_argument("--source", type=Path, default=SOURCE,
                   help="dump to cut the block from; any ID1,X1,ID2,X2,Y file")
    p.add_argument("--potency", choices=["nanomolar", "identity"],
                   default="nanomolar",
                   help="how Y becomes 'higher is better'. nanomolar = pIC50 "
                        "from IC50/nM (BindingDB); identity = already a potency "
                        "(KIBA score). Getting this wrong inverts the objective.")
    p.add_argument("--history", type=int, default=50,
                   help="free historical observations per auxiliary target")
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--acquisition", default="ei",
                   choices=[*ACQ_PORTFOLIO, "oracle", "switch_random",
                            "switch_rr"],
                   help="one of the fixed acquisitions; `oracle`, the per-round "
                        "hindsight ceiling over all of them; or the two free "
                        "switching controls, `switch_random` and `switch_rr`. "
                        "`oracle` READS THE TRUE VALUES and is a reference "
                        "line, not an arm: it says how much room an "
                        "acquisition chooser has before one is built, and the "
                        "controls say how much of that room costs nothing to "
                        "take. See FINDINGS 1")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history-cluster", type=float, default=0.0,
                   help="fraction of the history drawn from one tight chemical "
                        "neighbourhood instead of uniformly. 0 = same chemistry "
                        "as the search pool (reach ~1), 1 = a series of its own. "
                        "The achieved reach ratio is measured and reported")
    p.add_argument("--history-excludes-top", action="store_true",
                   help="drop the primary target's own top-k from the pool the "
                        "auxiliary history is drawn from. Tests whether the "
                        "campaign gain needs the history to point at the answer")
    p.add_argument("--arms", nargs="+", default=["stl", "mt", "pooled"],
                   choices=["stl", "mt", "pooled", "mt_r0", "stl_avoid",
                            "random", "similarity", "qed_only"],
                   help="which surrogates to run; stl alone is the budget ladder")
    p.add_argument("--report-druglike", type=float, default=None,
                   help="post-hoc scoring: search the FULL library, but count "
                        "only compounds above this QED as successes. The "
                        "counterpart to --qed-min, which removes them from the "
                        "library instead. Same success set, so the two are "
                        "comparable at equal absolute budget")
    p.add_argument("--audit", action="store_true",
                   help="check at runtime the invariants the comparison rests "
                        "on: equal initial design and shared history across "
                        "arms, exact budget spend, no repeat picks, no "
                        "eligibility violations, no primary-label leakage")
    p.add_argument("--restrict-selection", action="store_true",
                   help="with --report-druglike: spend the budget only on "
                        "compounds that can count as successes, while the "
                        "surrogate still gets free history from the whole "
                        "library. History is not charged, so cutting the "
                        "library up front discards it for no saving")
    p.add_argument("--multiobjective", action="store_true",
                   help="optimise potency AND drug-likeness as two objectives "
                        "(ParEGO selection, Pareto-front recall as the metric) "
                        "rather than potency alone. QED is never summed into "
                        "the reported score -- see FINDINGS 14 for why")
    p.add_argument("--subsample", type=int, default=None,
                   help="randomly cut the block to this many compounds; the "
                        "size control for --qed-min")
    p.add_argument("--qed-min", type=float, default=None,
                   help="drop compounds below this QED. A library admission "
                        "filter, not a score term, so the objective stays "
                        "entirely assayed and the floor stays at chance")
    p.add_argument("--sd-scale", type=float, default=1.0,
                   help="multiply the posterior standard deviation by this "
                        "before the acquisition reads it, leaving the mean "
                        "alone. The control for section 32: a multi-task arm "
                        "given pure noise still gains under EI because the "
                        "joint fit compresses sd, so a single-task arm at the "
                        "measured ratio should reproduce that gain with no "
                        "auxiliary data at all. 1.0 is a no-op")
    p.add_argument("--aux-shuffle", action="store_true",
                   help="permute every AUXILIARY column across compounds, "
                        "destroying the information while preserving the "
                        "marginal distribution exactly. The companion null to "
                        "--aux-quality 0, which preserves neither; the two "
                        "together separate the information in a column from "
                        "its shape")
    p.add_argument("--aux-permutation-scope", choices=["library", "history"],
                   default="library",
                   help="with --aux-shuffle: preserve the full library column "
                        "(legacy default) or the observed historical responses")
    p.add_argument("--aux-quality", type=float, default=None,
                   help="corrupt every AUXILIARY column to approximately this "
                        "Spearman correlation with its own true values, leaving "
                        "the primary target and the scoring untouched. Models a "
                        "*computed* auxiliary -- a docking score, a Boltz-2 "
                        "affinity -- in place of a measured one. 1.0 is the "
                        "measured panel every other section uses; the crossover "
                        "is the auxiliary-quality threshold, the multi-task "
                        "analogue of FINDINGS 8's prior-mean threshold of ~0.37")
    p.add_argument("--avoid-weight", type=float, default=None,
                   metavar="W",
                   help="multiply the acquisition value of every compound in "
                        "the auxiliary history by W, on the stl arm. 1.0 is a "
                        "no-op, 0.0 is near-total avoidance. Sweeping it "
                        "measures how much of the withheld-vs-uniform reversal "
                        "history avoidance alone can produce, at an avoidance "
                        "strength that can be matched to a model's own measured "
                        "selection ratio. A diagnostic, not a proposed method")
    p.add_argument("--scaffold-split", action="store_true",
                   help="draw the initial design only from compounds whose "
                        "Bemis-Murcko scaffold is absent from the true top-k, "
                        "so the campaign must cross scaffolds to succeed. The "
                        "random-split default lets a seed compound be a direct "
                        "analogue of the answer. Shorthand for --seed-pool "
                        "scaffold_split, kept because stored runs and "
                        "run_window_experiments pass it")
    p.add_argument("--seed-pool", choices=["random", "scaffold_split",
                                           "aux_top", "aux_scaffold_nn"],
                   default="random",
                   help="which compounds the initial design may be drawn from. "
                        "`random` is the whole library. `scaffold_split` is the "
                        "admission rule above. `aux_top` and `aux_scaffold_nn` "
                        "spend the free history on *where to start*, which no "
                        "arm here does -- the surrogate only sees it once the "
                        "first batch is bought. This is a property of the cell, "
                        "not of an arm: every arm in a run shares the pool, so "
                        "the comparison against `random` is between runs and "
                        "paired on --seed, exactly as --scaffold-split is")
    p.add_argument("--seed-top", type=int, default=10,
                   help="with --seed-pool aux_*: how many of each auxiliary's "
                        "best-scoring history compounds enter the pool")
    p.add_argument("--seed-nn", type=int, default=5,
                   help="with --seed-pool aux_scaffold_nn: Tanimoto neighbours "
                        "added per winner, on top of its scaffold-mates")
    p.add_argument("--pile-frac", type=float, default=0.02,
                   help="a value held by more than this fraction of a column is "
                        "treated as a reporting floor and its compounds dropped. "
                        "0.02 is the default used throughout; the threshold is a "
                        "heuristic and its sensitivity is swept in the supplement")
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--out", default="results/dense_block_campaign.json")
    args = p.parse_args(argv)
    if args.aux_shuffle and args.aux_quality is not None:
        raise SystemExit("--aux-shuffle and --aux-quality are two different "
                         "nulls; run them separately so each is reported "
                         "against its own baseline.")

    from molbo.surrogate import fingerprint_matrix, tanimoto_gram

    # Resolved before anything is read from disk: cutting a block takes seconds
    # to minutes, and a flag conflict should cost none of it.
    # --scaffold-split predates --seed-pool and is still passed by stored
    # invocations and by run_window_experiments, so it stays a valid spelling
    # of one mode rather than being replaced by it. Contradicting the two is an
    # error, not a silent precedence rule.
    seed_mode = args.seed_pool
    if args.scaffold_split:
        if seed_mode not in ("random", "scaffold_split"):
            p.error(f"--scaffold-split conflicts with --seed-pool {seed_mode}")
        seed_mode = "scaffold_split"

    permutation_provenance = []
    t0 = time.time()
    if args.blocks_from:
        from find_dense_blocks import blocks_as_families
        families = blocks_as_families(args.blocks_from, top=10_000)
    else:
        # PANEL_FAMILIES is merged in rather than added to FAMILIES, which three
        # sibling scripts import and pin their `choices` to. Those scripts read
        # the BindingDB dump only, and a panel family under a `--source` they
        # cannot change is an entry that is always wrong for them.
        families = {**FAMILIES, **PANEL_FAMILIES}
    if args.family not in families:
        p.error(f"unknown family {args.family!r}; choose from "
                + ", ".join(sorted(families)))
    mapping = families[args.family]
    # A panel family carries its own dump and its own scale. Filling them in is
    # not a convenience: the wrong `--potency` here inverts the objective
    # silently (see PANELS), and the default is the one that is wrong. Anything
    # the caller set explicitly is left alone and checked instead, because a
    # flag that overrides what the user typed is worse than the trap it closes.
    if args.family in PANEL_FAMILIES and not args.blocks_from:
        panel = PANELS[args.family.split("_")[0]]
        if args.source == SOURCE:
            args.source = panel["source"]
            print(f"[0] {args.family} is a {panel['source'].stem} block; "
                  f"--source {args.source}")
        if args.potency == p.get_default("potency"):
            args.potency = panel["potency"]
            print(f"[0] {args.family} reports a panel score; "
                  f"--potency {args.potency}")
        elif args.potency != panel["potency"]:
            p.error(f"{args.family} is a {panel['source'].stem} block and needs "
                    f"--potency {panel['potency']}; --potency {args.potency} "
                    "would invert the objective")
    potency = {"nanomolar": nanomolar_potency, "identity": lambda y: y}
    block, block_csv = build_block(mapping, source=args.source,
                                   potency=potency[args.potency])
    targets = [t for t in mapping.values() if t in block.columns]
    n_block = len(block)
    if not args.keep_censored:
        block, _ = drop_censored(block, targets, pile_frac=args.pile_frac)
    if args.qed_min is not None:
        block = drop_non_druglike(block, qed_min=args.qed_min)
    if args.subsample and args.subsample < len(block):
        # Control for the QED filter: a random subsample of the same size.
        # If unanimity degrades here too, the cause is pool size, not the
        # drug-likeness criterion.
        rs = np.random.default_rng(args.seed)
        block = block.iloc[rs.choice(len(block), size=args.subsample,
                                     replace=False)].reset_index(drop=True)
        print(f"\n[0c] random subsample to {len(block):,} compounds "
              "(size control for --qed-min)")
    # Filters can leave a block too small for the configured campaign. Without
    # this the failure surfaces as `a cannot be empty unless no samples are
    # taken` from inside the history draw, which names neither the flag that
    # caused it nor the number that was too small.
    # The history and the campaign draw from the same pool *independently* and
    # are allowed to overlap: a compound can carry a JAK1 historical value and
    # still be assayed on JAK2 by the campaign. That is not a leak, it is the
    # defining property of a dense block -- the auxiliary observations are on
    # the same molecules, which is why reach is 1.0 by construction (section
    # 18). So the pool must be large enough for the *largest single draw*, not
    # for their sum.
    #
    # An earlier version of this check summed them, and rejected configurations
    # that run correctly: CYP at 66% of the library is 221 + 50x3 + 10 = 381
    # against a 335-compound block, and it is a published row of the
    # operating-window table (section 25).
    need_hist = (args.history * (len(targets) - 1)
                 + (args.k if args.history_excludes_top else 0))
    hist_desc = (f"history {args.history}x{len(targets) - 1}"
                 + (f" drawn clear of the top-{args.k}"
                    if args.history_excludes_top else ""))
    # Keyed on the value only: max() over (value, string) tuples falls through
    # to comparing the strings whenever two needs tie, which made the message
    # depend on alphabetical order rather than on which constraint bound.
    needs = {f"a budget of {args.budget}": args.budget,
             hist_desc: need_hist,
             f"a top-{args.k}": args.k}
    what = max(needs, key=needs.get)
    need = needs[what]
    if len(block) < need:
        raise SystemExit(
            f"only {len(block)} compounds survive the filters, but {what} "
            f"needs at least {need} distinct ones. Relax "
            "--qed-min/--subsample, or lower --budget/--history.")

    X = fingerprint_matrix(block.smiles.tolist())
    K = tanimoto_gram(X, X)
    vals = block[targets].to_numpy(float)
    n_pool = len(block)
    print(f"{args.family}: {n_pool:,} compounds x {len(targets)} targets, "
          f"budget {args.budget}, batch {args.batch}, H={args.history} "
          f"({time.time() - t0:.0f}s)")

    arms = tuple(args.arms)
    qed_all = (compute_qed(block)
               if (args.multiobjective or args.report_druglike is not None)
               else None)
    qed = qed_all if args.multiobjective else None
    # Which compounds the budget may be spent on. `None` means all of them.
    # With --restrict-selection the surrogate still sees free history from the
    # whole library while the budget is confined to compounds that can count.
    eligible = (np.flatnonzero(qed_all >= args.report_druglike)
                if (args.report_druglike is not None and args.restrict_selection)
                else None)
    if args.report_druglike is not None:
        n_ok = int(np.sum(qed_all >= args.report_druglike))
        mode = ("budget confined to the" if args.restrict_selection
                else "whole library searchable;")
        print(f"    post-hoc scoring: {mode} {n_ok:,} drug-like compounds "
              f"(history spans all {len(block):,})")
    if qed is not None:
        print(f"    multi-objective: potency x QED, "
              f"QED {np.nanmin(qed):.2f}..{np.nanmax(qed):.2f}")
    # RDKit parses every SMILES, so once per block and only if asked for.
    scaf = (murcko_labels(block)
            if seed_mode in ("scaffold_split", "aux_scaffold_nn") else None)

    acc = {a: {"auc": [], "recall": [], "hist_ratio": []} for a in arms}
    achieved: list[float] = []
    reaches = []
    seed_sizes: list[int] = []
    seed_hits: list[int] = []
    audit = Audit(enabled=args.audit)
    digest = RunDigest()
    for ti, t in enumerate(targets):
        y = vals[:, ti]
        others = [j for j in range(len(targets)) if j != ti]
        if args.report_druglike is not None:
            # Post-hoc scoring: the campaign may assay anything, but only
            # drug-like compounds count as successes. The surrogate still
            # learns from a potent-but-undevelopable compound, which is the
            # information a pre-filter throws away; whether that information is
            # worth the assays it costs is exactly what this measures.
            #
            # The success set is identical to the one a pre-filtered run scores
            # against -- top-k potency among the drug-like -- so the two are
            # directly comparable at equal absolute budget.
            eligible = np.flatnonzero(qed_all >= args.report_druglike)
            order = eligible[np.argsort(y[eligible])]
            true_top = set(order[-args.k:].tolist())
            target_label = (f"top-{args.k} potency among {len(eligible)} "
                            f"drug-like (QED>={args.report_druglike:g})")
        elif qed is None:
            true_top = set(np.argsort(y)[-args.k:].tolist())
            target_label = f"top-{args.k}"
        else:
            # The success set is the true Pareto front over (potency, QED),
            # not a top-k of either objective alone.
            front = np.flatnonzero(pareto_mask(np.column_stack([y, qed])))
            true_top = set(front.tolist())
            target_label = f"Pareto front (|F|={len(front)})"
        seed_pool = None
        if seed_mode == "scaffold_split":
            seed_pool, n_banned, n_scaf = scaffold_disjoint_pool(
                block, y, args.k, np.random.default_rng(args.seed + ti),
                scaf=scaf)
            print(f"      {t}: scaffold split -- {len(seed_pool):,} of "
                  f"{len(block):,} compounds seedable "
                  f"({n_banned} of {n_scaf} scaffolds withheld)")
            if len(seed_pool) < args.n_init:
                raise SystemExit(
                    f"{t}: only {len(seed_pool)} compounds outside the top-k "
                    f"scaffolds, need {args.n_init} for the initial design")
        print(f"      {t}: success set = {target_label}", flush=True)
        for r in range(args.repeats):
            g = np.random.default_rng(args.seed + 1009 * r + 97 * ti)
            # History is drawn once per repeat and shared by every arm that
            # uses it, so a difference between arms is the surrogate and not a
            # luckier draw of free data.
            # The history may land on compounds that are themselves in the
            # primary's top-k. In deployment that is legitimate -- knowing a
            # compound's JAK1 number is real prior information, and you would
            # have it. But it means the auxiliary data can point straight at the
            # answer, and a reviewer will say so, so it has to be possible to
            # take it away and see whether the gain survives.
            hist = draw_history(K, y, args.history * len(others), g,
                                cluster_frac=args.history_cluster,
                                exclude_top=args.history_excludes_top, k=args.k)
            reaches.append(reach_ratio(K, hist))
            aux_idx = [hist[a * args.history:(a + 1) * args.history]
                       for a in range(len(others))]
            aux_val = [vals[aux_idx[a], others[a]] for a in range(len(others))]
            if args.aux_shuffle:
                # The distribution-preserving null. Same stream as
                # --aux-quality uses, so the two nulls see the same draw order
                # and differ only in what they do with the column.
                gq = np.random.default_rng(args.seed + 7717 * r + 131 * ti)
                aux_val, permutation_records = shuffle_auxiliary_responses(
                    vals, aux_idx, others, gq, args.aux_permutation_scope)
                achieved.extend(p["achieved_correlation"] for p in permutation_records)
                permutation_provenance.append({"target": t, "repeat": r,
                                               "permutations": permutation_records})
            if args.aux_quality is not None:
                # Redrawn per (target, repeat) so the answer averages over noise
                # draws rather than depending on one. The primary's own column
                # and the success set are untouched -- only what the multi-task
                # arm is told about its relatives degrades.
                gq = np.random.default_rng(args.seed + 7717 * r + 131 * ti)
                deg = []
                for a in range(len(others)):
                    full = degrade_to_rho(vals[:, others[a]], args.aux_quality, gq)
                    achieved.append(float(spearmanr(full, vals[:, others[a]])
                                          .statistic))
                    deg.append(full[aux_idx[a]])
                aux_val = deg
            if seed_mode.startswith("aux_"):
                # Inside the repeat loop, because the pool is a function of the
                # history draw -- unlike scaffold_split, which depends only on
                # the block. Computed once and handed to every arm, so the
                # initial design stays arm-independent and `audit.same_across`
                # still holds.
                seed_pool, seed_info = history_seed_pool(
                    K, scaf, aux_idx, aux_val, mode=seed_mode,
                    n_top=args.seed_top, n_nn=args.seed_nn)
                if len(seed_pool) < args.n_init:
                    raise SystemExit(
                        f"{t}: the {seed_mode} pool holds {len(seed_pool)} "
                        f"compounds, fewer than the {args.n_init} an initial "
                        "design needs. Raise --seed-top or --seed-nn.")
                seed_sizes.append(len(seed_pool))
                # How much of the answer the seeding rule was handed outright.
                # With --history-excludes-top this is 0 by construction; without
                # it, it is the number the gain has to be read against.
                seed_hits.append(len(set(np.argsort(y)[-args.k:].tolist())
                                     & set(seed_pool.tolist())))
            audit.cell()
            digest.cell(t, ti, r, aux_idx=aux_idx, aux_val=aux_val)
            # The denominator of the selection ratio below. Both are fixed for
            # this (target, repeat) cell, and every arm sees the same history,
            # so the ratio is comparable across arms by construction.
            hist_pool = (set(np.concatenate(aux_idx).tolist()) if aux_idx
                         else set())
            pool_n = n_pool if eligible is None else len(eligible)
            hist_in_pool = (hist_pool if eligible is None
                            else hist_pool & set(np.asarray(eligible).tolist()))
            hist_frac = (len(hist_in_pool) / pool_n) if pool_n else 0.0
            if hist_frac <= 0.0:
                hist_pool = set()
            for arm in arms:
                seq, picked_idx = run_campaign(
                    K, y, aux_idx, aux_val, arm=arm, budget=args.budget,
                    batch=args.batch, n_init=args.n_init, k=args.k,
                    acquisition=args.acquisition, qed=qed, eligible=eligible,
                    seed_pool=seed_pool, sd_scale=args.sd_scale,
                    avoid_weight=args.avoid_weight,
                    # Same stream per (repeat, target) so the initial design is
                    # identical across arms; only the surrogate differs.
                    rng=np.random.default_rng(args.seed + 1009 * r + 97 * ti))
                audit_campaign(picked_idx, arm=arm, budget=args.budget,
                               k=args.k, y=y, aux_idx=aux_idx,
                               eligible=eligible,
                               excl_top=args.history_excludes_top, audit=audit)
                # Arms must start from the same compounds and see the same
                # history; otherwise a difference between them is a luckier
                # draw rather than a better surrogate.
                audit.same_across((ti, r), arm, "initial design",
                                  picked_idx[:args.n_init])
                if arm == arms[0]:
                    # One arm is enough: `same_across` above fails the run if
                    # any other arm started somewhere else.
                    digest.init_design(t, ti, r, picked_idx[:args.n_init])
                audit.same_across((ti, r), arm, "history",
                                  np.concatenate(aux_idx) if aux_idx
                                  else np.empty(0, dtype=np.intp))
                # The history must never carry the primary target's own values
                # -- that would be the answer, relabelled. Pairwise: each
                # auxiliary is checked against the primary's values at its own
                # rows, not against one shared slice.
                for a_i, a_v in zip(aux_idx, aux_val):
                    audit.not_relabelled(y[a_i], [a_v], f"{arm} history")
                acc[arm]["auc"].append(auc_top_k(seq.tolist(), k=args.k))
                # By index, not by value: pIC50 ties are common in these blocks
                # and value membership would credit an arm for a compound it
                # never picked.
                acc[arm]["recall"].append(
                    len(true_top & set(picked_idx.tolist())) / args.k)
                # How hard this arm steered away from (or towards) the compounds
                # some relative had already been measured on. Enrichment, not a
                # raw share: the share alone rises with the history simply
                # because there is more of it to hit. 1.0 is indifference, 0.0
                # is total avoidance. Recorded per campaign so the avoidance
                # mechanism is readable off the artifact rather than needing a
                # second instrument over the same runs.
                if len(hist_pool):
                    bought = picked_idx[args.n_init:]
                    if len(bought):
                        share = len(set(bought.tolist()) & hist_pool) / len(bought)
                        acc[arm]["hist_ratio"].append(share / hist_frac)
        print(f"    {t:<8s} done", flush=True)

    audit.report("", clean_lines=(
        "all invariants hold: equal initial design, shared history,",
        "exact budget, no repeats, no eligibility or label violations",
    ))

    reach = float(np.mean(reaches)) if reaches else float("nan")
    print(f"\n    history reach ratio {reach:.2f} "
          f"(cluster_frac={args.history_cluster:g}; real ChEMBL libraries 0.53, "
          "co-measured block 1.00)")
    seeding = {"mode": seed_mode}
    if seed_sizes:
        hits = float(np.mean(seed_hits))
        seeding.update(pool_mean=float(np.mean(seed_sizes)),
                       pool_frac=float(np.mean(seed_sizes)) / n_pool,
                       top_k_in_pool_mean=hits,
                       seed_top=args.seed_top, seed_nn=args.seed_nn)
        print(f"    seeding {seed_mode}: pool {np.mean(seed_sizes):.0f} of "
              f"{n_pool:,} ({np.mean(seed_sizes) / n_pool:.1%}), "
              f"{hits:.1f} of the true top-{args.k} inside it"
              + (" -- history was drawn clear of the top-k, so this is 0 by "
                 "construction" if args.history_excludes_top and hits == 0
                 else " -- read the gain against this"))
    print(f"\n    {'arm':<10s}{'AUC top-' + str(args.k):>14s}{'recall':>12s}"
          f"{'vs stl':>16s}")
    print("    " + "-" * 52)
    base = np.asarray(acc["stl"]["auc"]) if "stl" in arms else None
    report = {"aux_permutation_scope": args.aux_permutation_scope if args.aux_shuffle else None,
              "aux_permutations": permutation_provenance,
              "family": args.family, "scaffold_split": bool(args.scaffold_split),
              "pile_frac": args.pile_frac,
              "history": args.history,
              "history_cluster": args.history_cluster, "reach": reach,
              "history_excludes_top": bool(args.history_excludes_top),
              "budget": args.budget, "batch": args.batch, "k": args.k,
              "n_init": int(args.n_init),
              # Every RNG stream in the run is `seed + f(repeat, target)`, so
              # this is the whole of what makes two runs comparable draw for
              # draw. It was the one thing a stored artefact could not answer
              # when two runs of the same cell had to be told apart.
              "seed": int(args.seed),
              "acquisition": args.acquisition, "repeats": args.repeats,
              # The acquisition's own hyperparameters, read off the module that
              # supplies them rather than retyped: `run_campaign` never passes
              # either, so what a campaign ran with *is* the library default,
              # and a stored literal would start lying the day that moves.
              "xi": float(EI_XI), "beta": float(BETA),
              "avoid_weight": (None if args.avoid_weight is None
                               else float(args.avoid_weight)),
              "sd_scale": float(args.sd_scale),
              "n_pool": int(n_pool), "seeding": seeding,
              "provenance": block_provenance(
                  args, targets=targets, block_csv=block_csv,
                  n_block=n_block, n_pool=n_pool),
              "arms": {}}
    if achieved:
        # The requested rho is a Pearson target; what the auxiliary actually
        # ended up correlating at is the number the threshold is stated in.
        # Under --aux-shuffle there is no request, only an achieved value, and
        # it should sit at zero to within the sampling error of the column.
        a_mean = float(np.mean(achieved))
        print(f"\n    auxiliary quality: "
              f"requested {args.aux_quality if args.aux_quality is not None else 'shuffled'}, "
              f"achieved Spearman {a_mean:.3f} "
              f"({np.min(achieved):.3f}..{np.max(achieved):.3f} over "
              f"{len(achieved)} draws)")
        report["aux_quality_requested"] = (
            float(args.aux_quality) if args.aux_quality is not None
            else "shuffled")
        report["aux_quality_achieved"] = a_mean

    # Per-target, not just pooled. A practitioner cannot measure the gain on the
    # cold target they care about -- it has no data yet. What they can do is
    # hold out each target they *do* have and check whether multi-task wins
    # there. That self-check is only usable if the answer is consistent across
    # targets, so the spread across them is the quantity that decides whether
    # the procedure is a guideline or a coin flip.
    if "mt" in arms and "stl" in arms:
        print(f"\n    per-target (the leave-one-target-out self-check)")
        print(f"    {'held-out target':<18s}{'stl':>8s}{'mt':>8s}{'delta recall':>15s}")
        per = {}
        for ti, t in enumerate(targets):
            lo, hi = ti * args.repeats, (ti + 1) * args.repeats
            s = float(np.mean(acc["stl"]["recall"][lo:hi]))
            m = float(np.mean(acc["mt"]["recall"][lo:hi]))
            per[t] = {"stl_recall": s, "mt_recall": m, "delta": m - s}
            print(f"    {t:<18s}{s:>8.1%}{m:>8.1%}{m - s:>+15.1%}")
        d = np.asarray([v["delta"] for v in per.values()])
        agree = int((d > 0).sum())
        print(f"    {agree}/{len(targets)} targets improve; spread "
              f"{d.min():+.1%} to {d.max():+.1%}")
        if agree == len(targets):
            print("    -> unanimous: the self-check is informative on this panel.")
        else:
            print("    -> not unanimous: a leave-one-out check on this panel would")
            print("       give a different answer depending on which target you held")
            print("       out, so it cannot be used to decide for a new one.")
        report["per_target"] = per
        report["targets_improving"] = agree

    # The audit verdict travels with the numbers it vouches for. A table whose
    # JSON says `clean: false` is not quotable, and the run that produced the
    # operating-window table had no record of having been audited at all --
    # which is how it went into a paper unaudited. See FINDINGS section 30.
    report["audit"] = {**audit.to_json(), "arms_per_cell": len(arms)}
    report["digest"] = digest.to_json()

    for arm in arms:
        a = np.asarray(acc[arm]["auc"])
        rec = np.asarray(acc[arm]["recall"])
        # `--arms` need not contain `stl`: a sweep that only wants the multi-task
        # trace is a legitimate run, and `base` is then None. Every delta below
        # is *against* stl, so without it there is no reference and the honest
        # value is null rather than zero. Writing 0.0 would be worse than the
        # TypeError this replaces -- a reader, or `acquisition_across_families`,
        # would see `delta_vs_stl: 0.0` and read a measured tie where nothing
        # was measured at all.
        d = None if base is None else a - base
        if d is None:
            delta = "n/a"
        elif arm == "stl":
            delta = ""
        else:
            delta = f"{d.mean():>+10.4f}+-{d.std() / np.sqrt(len(d)):<5.4f}"
        print(f"    {arm:<10s}{a.mean():>14.4f}{rec.mean():>12.1%}{delta:>16s}")
        hr = np.asarray(acc[arm]["hist_ratio"], dtype=float)
        report["arms"][arm] = {
            # 1.0 = picks history at exactly its share of the pool, 0.0 = never.
            "history_selection_ratio": (float(hr.mean()) if hr.size else None),
            "history_selection_ratio_sem": (
                float(hr.std() / np.sqrt(hr.size)) if hr.size else None),
            "auc_mean": float(a.mean()),
            "auc_sem": float(a.std() / np.sqrt(len(a))),
            "recall_mean": float(rec.mean()),
            "delta_vs_stl": None if d is None else float(d.mean()),
            "delta_sem": (None if d is None
                          else float(d.std() / np.sqrt(len(d))) if arm != "stl"
                          else 0.0),
            "frac_better": (None if d is None
                            else float((d > 0).mean()) if arm != "stl" else 0.0),
            # The per-campaign traces, in (target, repeat) order, identical
            # across arms because they share the initial design and history.
            # Without them the only comparisons available from the artefact are
            # each arm against `stl`, and a question like "does the fitted ICM
            # beat naive pooling" then has to be answered by differencing two
            # independent SEMs -- which is not the paired test it needs and
            # cost this project a wrong answer once already (section 31).
            "auc": [float(x) for x in a],
            "recall": [float(x) for x in rec],
        }

    if "mt" not in arms or "stl" not in arms:
        # A budget ladder runs stl alone; there is no gain to adjudicate.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
        return 0
    mt = report["arms"]["mt"]
    pooled = report["arms"].get("pooled", {"delta_vs_stl": -np.inf})
    print("\n    verdict:")
    if mt["delta_vs_stl"] > 3 * mt["delta_sem"]:
        print(f"      the transfer gain survives as a campaign result: "
              f"{mt['delta_vs_stl']:+.4f} AUC")
        print(f"      over single-task BO ({mt['frac_better']:.0%} of campaigns "
              "better).")
        if pooled["delta_vs_stl"] > mt["delta_vs_stl"] - 3 * mt["delta_sem"]:
            print("      But pooling matches it -- on this block the campaign gain")
            print("      does not require modelling the task structure.")
    elif mt["delta_vs_stl"] > 0:
        # Not the same verdict as a negative result, and it used to print as
        # one. A cell at +0.081 +- 0.039 is 2.1 SEM positive; calling that "no
        # campaign gain" is how an underpowered cell gets quoted as a null.
        # Section 34 runs at 8 repeats per cell, where this branch is the
        # common case rather than the exception.
        print(f"      not established at this power: {mt['delta_vs_stl']:+.4f}"
              f"+-{mt['delta_sem']:.4f} AUC, which is "
              f"{mt['delta_vs_stl'] / max(mt['delta_sem'], 1e-12):.1f} SEM and")
        print("      the bar here is 3. The point estimate is positive, so this")
        print("      is a cell to re-run with more repeats, not a null.")
    else:
        print(f"      no campaign gain: {mt['delta_vs_stl']:+.4f}"
              f"+-{mt['delta_sem']:.4f} AUC. The regression improvement in")
        print("      sections 18-19 does not convert into finding the top faster.")
        print("      That is the section 4 pattern again: an offline proxy moving")
        print("      without the campaign outcome moving.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""If a chooser could abstain, would a region prior pay? -- the calibration axis.

    python scripts/selective_headroom.py --n-targets 24 --budget 30 --seeds 200

FINDINGS section 4 ends on a number that closes the region-prior line: measured
over 24 random held-out targets the break-even accuracy is **above 100%**, so no
top-1 accuracy suffices. The mechanism given there is not accuracy at all --

    "the problem is not the chooser's accuracy but committing at all"

-- and that sentence contains an untested assumption. Every chooser in this
project is *forced to act*. It names a region every time it is asked, including
on the targets where region restriction is worth nothing (46% of them, section
3). A chooser that reported calibrated confidence could decline, and a declined
round costs exactly what the free bandit costs.

So this measures the axis the project has never varied: **how much does a
chooser have to know about its own correctness**, and is that reachable?

Four quantities, in increasing order of what they assume:

``bandit``               the free control. The bar, as in section 4.
``commit(prior)``        act on the prior's pick always. Reproduces section 4.
``gate(correctness)``    act iff the pick is the best region -- the ceiling for
                         a confidence signal that tracks *being right*.
``gate(outcome)``        act iff acting would have beaten the bandit on this
                         target. Reads the campaign answer key, so it bounds
                         **every possible abstention rule**, calibrated or not.
                         If this is small, the whole gated-chooser line is dead
                         and no amount of calibration work changes it.

Then two things that are not ceilings:

1. **Real confidence.** Several signals computable with no answer key are
   scored by their AUROC against "acting here beats the bandit". The one that
   matters is ``pred_lift``: not *which* region is best, but whether this
   target's predicted actives are concentrated at all. Section 3 found the
   hindsight lift predicts the ceiling at Spearman +0.602; ``pred_lift`` is its
   answer-key-free twin, and it asks a Noul-shaped question -- *is region
   restriction worth doing on this target* -- rather than a Choice-shaped one.

2. **The break-even surface.** A synthetic chooser is swept over accuracy alpha
   (P(pick = best region)) and confidence discrimination delta (AUROC of its
   confidence against its own correctness). delta = 0.5 is an uncalibrated
   chooser, which is every chooser this project has built. The output is the
   contour where gating starts to pay, in (alpha, delta) rather than in alpha
   alone.

Cost discipline: the campaigns are run **once** per (target, region) and every
policy above is algebra over the stored per-seed AUC. A sweep of 5 policies x
10 accuracies x 6 discriminations costs the same as one sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ESM2 = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
        "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties at 0.5.

    n is one point per target here, so 24 -- this is reported alongside its own
    permutation null rather than read as a calibrated probability.
    """
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def auroc_p(score, label, draws=20000, seed=0):
    """Permutation p-value for an AUROC with this few points."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    obs = auroc(score, label)
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = np.array([auroc(score, rng.permutation(label)) for _ in range(draws // 100)])
    return float(np.mean(np.abs(null - 0.5) >= abs(obs - 0.5)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-targets", type=int, default=24)
    p.add_argument("--min-compounds", type=int, default=250)
    p.add_argument("--max-compounds", type=int, default=1200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--min-candidates", type=int, default=10)
    p.add_argument("--esm", default="35M", choices=list(ESM2))
    p.add_argument("--n-train", type=int, default=150_000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0,
                   help="target draw. 0 reproduces the 24 targets of section 4.")
    p.add_argument("--acquisition", default="ei", choices=["ei", "ucb", "pi", "greedy"],
                   help="Not a detail. Section 9 measured that a prior and an "
                        "acquisition function cannot be chosen independently -- a "
                        "PERFECT prior scores -0.019 to -0.144 under EI and +0.057 "
                        "to +0.113 under UCB, because a confident prior collapses "
                        "the posterior sd and EI then ties across the library. "
                        "Every abstention number in sections 12-13 is under EI, so "
                        "whether the gate's ceiling survives UCB is an open "
                        "question this flag exists to answer.")
    p.add_argument("--offset", type=int, default=0,
                   help="skip this many targets of the draw before taking "
                        "--n-targets. Lets one permutation be tiled across "
                        "parallel processes without overlap, so the eligible "
                        "pool can be covered exhaustively rather than sampled.")
    p.add_argument("--sim-draws", type=int, default=4000,
                   help="Monte-Carlo draws per (alpha, delta) cell")
    p.add_argument("--replay", default=None,
                   help="re-run the analysis from a previous run's JSON without "
                        "touching a campaign. Everything after the sweep is "
                        "algebra over stored per-seed AUC, so a new policy costs "
                        "seconds rather than an hour.")
    p.add_argument("--out", default="results/selective_headroom.json")
    return p.parse_args(argv)


def fit_prior(df, test, args):
    """The section-4 prior: ridge on (ESM2 protein embedding, ECFP4), fitted on
    every target not under test. Returns a callable (tid, library) -> pIC50."""
    from molbo.clm import Encoder
    from molbo.surrogate import fingerprint_matrix

    size = df.groupby("target_id").size()
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    train = df[~df.target_id.isin(set(test))]
    train = train[train.target_id.isin(size[(size >= 200) & (size <= 3000)].index)]
    train = train[[len(seqs[t]) <= 1022 for t in train.target_id]]
    enc = Encoder(model=ESM2[args.esm], revision=None, pooling="mean", max_len=1024,
                  cache_dir="data/cache/esm")
    uniq = sorted(set(train.target_id) | set(test))
    P = dict(zip(uniq, enc.embed([seqs[t] for t in uniq], progress=True)))
    tr = train.sample(n=min(args.n_train, len(train)), random_state=0)
    t0 = time.time()
    Xtr = np.hstack([np.stack([P[t] for t in tr.target_id]).astype(np.float32),
                     fingerprint_matrix(list(tr.smiles)).astype(np.float32)])
    ytr = tr.pic50.to_numpy()
    mu, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    G = A.T @ A
    G[np.diag_indices_from(G)] += args.alpha
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    print(f"[gate] ridge on {Xtr.shape} from {tr.target_id.nunique()} targets "
          f"in {time.time() - t0:.0f}s\n", flush=True)

    def predict(tid, lib):
        X = np.hstack([np.repeat(P[tid][None, :], len(lib), 0).astype(np.float32),
                       fingerprint_matrix(lib).astype(np.float32)])
        return (X - mu) @ w + ym

    return predict


def region_scores(pred, labels, k):
    """Section 4's region score: the share of a region that the model places in
    the library's predicted top 1%. Returns the scores and every confidence
    feature that can be read off them with no answer key."""
    thr = np.quantile(pred, 0.99)
    sizes = np.bincount(labels, minlength=k).astype(float)
    hits = np.array([(pred[labels == c] >= thr).sum() for c in range(k)], float)
    score = np.where(sizes > 0, hits / np.maximum(sizes, 1.0), -1.0)
    order = np.argsort(-score)
    p = np.clip(score, 0, None)
    q = p / p.sum() if p.sum() > 0 else np.full(k, 1.0 / k)
    n_top = max(1.0, hits.sum())
    pick = int(order[0])
    return score, {
        "margin": float(score[order[0]] - score[order[1]]),
        "top1_score": float(score[order[0]]),
        "entropy": float(-(q * np.log(q + 1e-12)).sum() / np.log(k)),
        "n_pred_top_in_pick": float(hits[pick]),
        # the answer-key-free twin of section 3's `best_lift`: how concentrated
        # the model thinks this target's actives are. A Noul, not a Choice.
        "pred_lift": float((hits[pick] / n_top) / max(sizes[pick] / sizes.sum(), 1e-9)),
    }


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)

    if args.replay:
        saved = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        rows = saved["rows"]
        for r in rows:  # JSON turns the region keys into strings
            r["delta_vs_bandit"] = {int(c): v for c, v in r["delta_vs_bandit"].items()}
        args.seeds, args.k = saved["seeds"], saved["k"]
        print(f"[gate] replaying {len(rows)} targets from {args.replay}")
        analyse(rows, {}, args)
        return 0

    from molbo.campaign import ArmConfig
    from molbo.clusters import FixedClusterSource, UCBClusterSource, cluster_library, concentration
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle
    from prior_sweep import load_pairs

    df = load_pairs()
    size = df.groupby("target_id").size()
    seqs = df.drop_duplicates("target_id").set_index("target_id").sequence
    pool = [t for t in size[(size >= args.min_compounds)
                            & (size <= args.max_compounds)].index
            if len(seqs[t]) <= 1022]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))[args.offset:args.offset + args.n_targets]
    test = [pool[i] for i in order]
    print(f"[gate] {len(pool)} eligible targets; testing {len(test)} "
          f"(draw seed {args.seed}, offset {args.offset}, k={args.k}, "
          f"budget {args.budget}, {args.seeds} seeds)", flush=True)

    predict = fit_prior(df, test, args)

    tmp = Path("data/cache/sweep_libs")
    tmp.mkdir(parents=True, exist_ok=True)
    rows, per_seed = [], {}
    hdr = (f"{'target':<10s}{'n':>5s}{'pick':>5s}{'best':>5s}{'':>5s}"
           f"{'lift':>7s}{'pLift':>7s}{'bandit':>9s}{'commit':>9s}{'d_best':>9s}")
    print(hdr + "\n" + "-" * len(hdr), flush=True)

    for tid in test:
        sub = df[df.target_id == tid]
        csv = tmp / f"{tid}.csv"
        sub.rename(columns={"smiles": "canonical_smiles", "pic50": "pchembl_value"})[
            ["canonical_smiles", "pchembl_value"]].to_csv(csv, index=False)
        try:
            oracle = LookupOracle.from_dataset(csv, budget=args.budget)
        except Exception:
            continue
        lib = list(oracle.library)
        if len(lib) < args.k * 8:
            continue
        values = np.array([oracle._table[s] for s in lib])
        labels = cluster_library(lib, args.k, method="kmeans")
        k = int(labels.max()) + 1
        conc = concentration(labels, values, 0.01)
        best = int(conc["best_cluster"])
        pred = predict(tid, lib)
        score, conf = region_scores(pred, labels, k)
        pick = int(np.argmax(score))

        arms = [ArmConfig("bo_whole", strategy="bo", acquisition=args.acquisition, n_init=args.n_init),
                ArmConfig("bandit", strategy="bo", acquisition=args.acquisition, n_init=args.n_init,
                          min_candidates=args.min_candidates, constraint_every=1,
                          constraint_source=UCBClusterSource(labels=labels))]
        for c in range(k):
            if (labels == c).sum() == 0:
                continue
            arms.append(ArmConfig(f"commit{c}", strategy="bo", acquisition=args.acquisition,
                                  n_init=args.n_init, min_candidates=args.min_candidates,
                                  constraint_source=FixedClusterSource(labels, c)))
        m, _ = run_sweep(oracle, arms, range(args.seeds), budget=args.budget, k=10,
                         library=lib, verbose=False)
        by = {}
        for a in arms:
            v = {x.seed: x.auc for x in m if x.arm == a.name}
            by[a.name] = np.array([v[s] for s in range(args.seeds)])
        per_seed[tid] = by

        band = by["bandit"]
        d_commit = {c: float((by[f"commit{c}"] - band).mean()) for c in range(k)
                    if f"commit{c}" in by}
        rows.append({
            "target_id": tid, "n": len(lib), "k": k, "pick": pick, "best": best,
            "hit": pick == best, "best_lift": float(conc["best_lift"]),
            "sizes": np.bincount(labels, minlength=k).tolist(),
            "conf": conf,
            "delta_vs_bandit": d_commit,
            "bandit_minus_whole": float((band - by["bo_whole"]).mean()),
            "region_scores": score.tolist(),
        })
        print(f"{tid:<10s}{len(lib):>5d}{pick:>5d}{best:>5d}"
              f"{'  HIT' if pick == best else ' miss':>5s}"
              f"{conc['best_lift']:>7.1f}{conf['pred_lift']:>7.1f}"
              f"{rows[-1]['bandit_minus_whole']:>+9.4f}"
              f"{d_commit.get(pick, float('nan')):>+9.4f}"
              f"{d_commit.get(best, float('nan')):>+9.4f}", flush=True)

    analyse(rows, per_seed, args)
    return 0


# --------------------------------------------------------------------------
# Everything below is algebra over the stored per-seed AUC -- no new campaigns.
# --------------------------------------------------------------------------

def analyse(rows, per_seed, args):
    from scipy.stats import norm, spearmanr, wilcoxon

    T = len(rows)
    d_pick = np.array([r["delta_vs_bandit"][r["pick"]] for r in rows])
    d_best = np.array([r["delta_vs_bandit"][r["best"]] for r in rows])
    hit = np.array([r["hit"] for r in rows], bool)

    print(f"\n\n{'=' * 74}\nGATING A CHOOSER THAT MUST vs MAY ACT  ({T} targets, "
          f"{args.seeds} seeds)\n{'=' * 74}")
    print("\nreference: the free UCB bandit over regions. All deltas are vs it.\n")

    def line(name, d, note=""):
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  {name:<34s}{d.mean():>+9.4f}{d.std(ddof=1) / np.sqrt(T):>9.4f}"
              f"{p:>9.4f}   {note}")

    print(f"  {'policy':<34s}{'dAUC':>9s}{'sem':>9s}{'p':>9s}")
    print("  " + "-" * 68)
    line("commit(prior)          [sec 4]", d_pick, "acts on every target")
    line("gate(correctness)", np.where(hit, d_pick, 0.0), f"acts on {hit.mean():.0%}")
    gate_out = np.maximum(d_pick, 0.0)
    line("gate(outcome)   CEILING", gate_out, f"acts on {(d_pick > 0).mean():.0%}")
    line("commit(best region)    [sec 3]", d_best, "reads the answer key")
    line("gate(outcome) on best region", np.maximum(d_best, 0.0), "both answer keys")

    # ---- real, answer-key-free confidence signals ------------------------
    print("\n\nDoes anything computable predict when acting pays?")
    print(f"  {'signal':<24s}{'AUROC vs pays':>15s}{'perm p':>9s}"
          f"{'AUROC vs hit':>14s}{'rho vs dAUC':>13s}")
    print("  " + "-" * 75)
    pays = d_pick > 0
    feats = sorted(rows[0]["conf"])
    conf_auroc = {}
    for f in feats:
        v = np.array([r["conf"][f] for r in rows], float)
        a1, a2 = auroc(v, pays), auroc(v, hit)
        pv = auroc_p(v, pays)
        rho = float(spearmanr(v, d_pick).statistic)
        conf_auroc[f] = {"auroc_pays": a1, "perm_p": pv, "auroc_hit": a2,
                         "spearman_dauc": rho}
        print(f"  {f:<24s}{a1:>15.3f}{pv:>9.3f}{a2:>14.3f}{rho:>+13.3f}")
    v = np.array([r["best_lift"] for r in rows], float)
    print(f"  {'best_lift [answer key]':<24s}{auroc(v, pays):>15.3f}"
          f"{auroc_p(v, pays):>9.3f}{auroc(v, hit):>14.3f}"
          f"{float(spearmanr(v, d_pick).statistic):>+13.3f}")

    # ---- the best real gate, honestly evaluated --------------------------
    print("\n\nA gate on each real signal, thresholded")
    print(f"  {'signal':<18s}{'rule':<24s}{'acts on':>9s}{'dAUC':>10s}{'p':>9s}")
    print("  " + "-" * 70)
    gates = {}
    for f in feats:
        v = np.array([r["conf"][f] for r in rows], float)
        best_tau, best_d, best_cov = (0.0, 1), -np.inf, 0.0
        for tau in np.unique(v):
            for sign in (1, -1):
                act = (v >= tau) if sign > 0 else (v <= tau)
                if act.sum() == 0:
                    continue
                dd = float(np.where(act, d_pick, 0.0).mean())
                if dd > best_d:
                    best_d, best_tau, best_cov = dd, (float(tau), sign), float(act.mean())
        # leave-one-out: the threshold is chosen without the target it scores
        loo = np.zeros(T)
        for i in range(T):
            idx = np.arange(T) != i
            bt, bd = (0.0, 1), -np.inf
            for tau in np.unique(v[idx]):
                for sign in (1, -1):
                    act = (v[idx] >= tau) if sign > 0 else (v[idx] <= tau)
                    if act.sum() == 0:
                        continue
                    dd = float(np.where(act, d_pick[idx], 0.0).mean())
                    if dd > bd:
                        bd, bt = dd, (float(tau), sign)
            act_i = (v[i] >= bt[0]) if bt[1] > 0 else (v[i] <= bt[0])
            loo[i] = d_pick[i] if act_i else 0.0
        p = wilcoxon(loo).pvalue if np.any(loo != 0) else 1.0
        gates[f] = {"tau": best_tau[0], "sign": best_tau[1],
                    "in_sample_dauc": best_d, "coverage": best_cov,
                    "loo_dauc": float(loo.mean()), "loo_p": float(p)}
        print(f"  {f:<18s}{'in-sample (optimistic)':<24s}{best_cov:>8.0%}"
              f"{best_d:>+10.4f}{'':>9s}")
        print(f"  {'':<18s}{'leave-one-out (honest)':<24s}{'':>8s}"
              f"{loo.mean():>+10.4f}{p:>9.4f}")

    # ---- bootstrap the headline policies over targets --------------------
    brng = np.random.default_rng(7)
    print("\n\n95% bootstrap interval over targets (10k resamples)")
    boots = {}
    for nm, d in (("commit(prior)", d_pick),
                  ("gate(correctness)", np.where(hit, d_pick, 0.0)),
                  ("gate(outcome) CEILING", gate_out),
                  ("commit(best region)", d_best)):
        idx = brng.integers(T, size=(10000, T))
        bs = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boots[nm] = [float(lo), float(hi)]
        print(f"  {nm:<26s}{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")

    # ---- the break-even surface in (accuracy, calibration) ---------------
    alphas = np.round(np.arange(0.1, 1.001, 0.1), 3)
    deltas = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    wrong = [np.array([val for c, val in r["delta_vs_bandit"].items() if c != r["best"]])
             for r in rows]

    def surface_for(mode, seed):
        """dAUC over an (alpha, delta) grid.

        ``mode`` is what the gate's confidence discriminates:

        ``hit``   whether the pick is the best region -- the natural reading of
                  "calibrated", and the one a Choice-shaped question produces.
        ``pays``  whether acting on the pick beats the bandit on this target.
                  A strictly different label, and the measured signals separate
                  it far better (AUROC 0.72 vs 0.55). Gating is a decision about
                  the *outcome*, so this is the better-matched target and the
                  one a Noul-shaped question produces.
        """
        srng = np.random.default_rng(seed)
        D = args.sim_draws
        surf = np.zeros((len(alphas), len(deltas)))
        cov = np.zeros_like(surf)
        for ia, a in enumerate(alphas):
            correct = srng.random((D, T)) < a
            draw_wrong = np.stack([w[srng.integers(len(w), size=D)] for w in wrong], 1)
            dv = np.where(correct, d_best[None, :], draw_wrong)
            label = correct if mode == "hit" else (dv > 0)
            for id_, dl in enumerate(deltas):
                sep = np.sqrt(2.0) * norm.ppf(min(max(dl, 0.5), 0.999999))
                c = srng.normal(0, 1, (D, T)) + sep * label
                bestv, bestc = -np.inf, 0.0
                for q in np.arange(0.0, 1.0, 0.05):
                    tau = -np.inf if q == 0 else np.quantile(c, q)
                    act = c >= tau
                    val = float(np.where(act, dv, 0.0).mean())
                    if val > bestv:
                        bestv, bestc = val, float(act.mean())
                surf[ia, id_], cov[ia, id_] = bestv, bestc
        return surf, cov

    surfaces = {}
    for mode, blurb in (("hit", "confidence tracks WHICH REGION IS RIGHT "
                                "(a Choice-shaped question)"),
                        ("pays", "confidence tracks WHETHER ACTING HELPS "
                                 "(a Noul-shaped question)")):
        surf, cov = surface_for(mode, 1234)
        surfaces[mode] = {"dauc": surf.tolist(), "coverage": cov.tolist()}
        print(f"\n\nBREAK-EVEN SURFACE ({mode}): dAUC vs the free bandit")
        print(f"  alpha = P(pick = best region).  delta = AUROC of its confidence")
        print(f"  against {mode}: {blurb}.")
        print("  delta = 0.50 is an uncalibrated chooser, which is every chooser")
        print("  this project has built. The threshold is chosen per cell to")
        print("  maximise dAUC -- optimistic for the gate, and it still has to")
        print("  clear zero.\n")
        print("  alpha |" + "".join(f"{d:>10.2f}" for d in deltas))
        print("  ------+" + "-" * (10 * len(deltas)))
        for ia, a in enumerate(alphas):
            print(f"  {a:>5.1f} |" + "".join(f"{surf[ia, i]:>+10.4f}"
                                             for i in range(len(deltas))))
        print("\n  coverage (fraction of targets acted on) at the optimising threshold")
        print("  alpha |" + "".join(f"{d:>10.2f}" for d in deltas))
        print("  ------+" + "-" * (10 * len(deltas)))
        for ia, a in enumerate(alphas):
            print(f"  {a:>5.1f} |" + "".join(f"{cov[ia, i]:>10.0%}"
                                             for i in range(len(deltas))))
        print("\n  break-even accuracy (lowest alpha with dAUC > 0), per delta")
        for id_, dl in enumerate(deltas):
            ok = np.flatnonzero(surf[:, id_] > 0)
            s = f"{alphas[ok[0]]:.0%}" if len(ok) else ">100%"
            print(f"    delta {dl:.2f}  ->  {s}")
        # the dual reading: hold accuracy at what the prior measures, and ask
        # what the gate would have to be worth.
        a_meas = float(hit.mean())
        ia = int(np.argmin(np.abs(alphas - a_meas)))
        row = surf[ia]
        ok = np.flatnonzero(row > 0)
        need = f"{deltas[ok[0]]:.2f}" if len(ok) else ">1.00"
        best_real = max(v["auroc_pays" if mode == "pays" else "auroc_hit"]
                        for v in conf_auroc.values())
        print(f"\n  at the prior's MEASURED accuracy ({a_meas:.0%}, grid "
              f"{alphas[ia]:.0%}): a gate needs delta >= {need}")
        print(f"  the best free signal measured here reaches delta = {best_real:.3f}"
              f"  ->  {'CLEARS IT' if str(need) != '>1.00' and best_real >= float(need) else 'short'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "budget": args.budget, "seeds": args.seeds, "k": args.k,
        "n_targets": T, "accuracy": float(hit.mean()),
        "policies": {
            "commit_prior": float(d_pick.mean()),
            "gate_correctness": float(np.where(hit, d_pick, 0.0).mean()),
            "gate_outcome_ceiling": float(gate_out.mean()),
            "commit_best_region": float(d_best.mean()),
            "gate_outcome_on_best": float(np.maximum(d_best, 0.0).mean()),
        },
        "confidence_auroc": conf_auroc, "gates": gates, "bootstrap": boots,
        "surface": {"alphas": alphas.tolist(), "deltas": deltas.tolist(),
                    **surfaces},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    if per_seed:
        np.savez_compressed(out.with_suffix(".npz"),
                            **{f"{t}|{a}": v for t, by in per_seed.items()
                               for a, v in by.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())

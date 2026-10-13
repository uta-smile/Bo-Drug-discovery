"""The shortlist hook's missing ablation: can the gate gamma be SUPPLIED?

    python scripts/shortlist_gate.py --datasets data/chembl/cruzipain.csv ...

Section 13 sweeps a *synthetic* chooser over (alpha, gamma) and finds that any
gamma at all collapses the break-even accuracy from 20% to chance. That is a
statement about what a gate would be worth, not about whether one exists.
`scripts/gated_selector.py` then shows no hand-written run-time rule beats a
random gate firing equally often. This closes the loop: it trains the gate and
measures the gamma it actually achieves.

Why this hook admits a clean answer where the region hook does not: the gate's
label is **observable every round**. Section 13 defines

    gamma = P(act | right) - P(act | wrong)

where "right" means the chooser named the shortlist's truly best compound --
and on a retrospective library that is known the moment the shortlist is drawn.
No counterfactual rollout, no campaign-outcome label, ~20-50 labelled decisions
per campaign instead of one per target. For a binary gate gamma is exactly
Youden's J, and a scored gate's AUROC maps to it as gamma_max = max_tau J(tau).

The chooser under test is ``MaxMeanSelector`` -- buy the highest posterior
mean. It is the project's sharpest case of being good at the wrong thing (3.4x
plain BO's offline gain, negative campaign AUC), and "when is the posterior
argmax actually the best compound here" is precisely the question a gate on it
has to answer.

Design, and the two things it is careful about:

**Training data matches deployment.** Records come from ``RecordingSelector``,
which returns rank 0 -- so the harvesting run *is* plain BO and the shortlists
are the EI top-k lists a deployed gate would face, not a distribution shifted
by the gate's own past actions.

**Leave-one-library-out.** Rounds within a campaign are correlated and
campaigns within a library share a compound pool, so a random split over
decisions would leak badly. The gate is trained on other libraries entirely,
which is also the honest transfer question: would this gate work on a target it
has never seen?

The offline gamma is then **not** trusted on its own. Section 4 recorded two
occasions where an offline shortlist metric inverted against campaign outcome,
so the achieved gate is wired back into a real campaign arm and the paired
ΔAUC is reported next to it. Any disagreement between the two is the finding.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

FEATURES = [
    "lead_mu",          # how far the pick's posterior mean beats rank 0's
    "lead_over_sd",     # the same, in posterior sds -- a natural confidence
    "mu_std",           # spread of the shortlist's means
    "sd_mean",          # how much the GP still does not know here
    "sd_at_pick",
    "sd_at_rank0",
    "sd_ratio",         # pick's uncertainty relative to the shortlist
    "acq_lead",         # EI's own preference for rank 0 over the pick
    "acq_std",
    "round_frac",       # position in the campaign
    "n_obs",
    "best_obs",         # best value measured so far -- observed, not answer key
    "obs_spread",
    "shortlist_n",
]


def features(mu, sd, acq, observations, round_index, n_rounds) -> np.ndarray:
    """Everything a gate could see at decision time. No answer key."""
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    acq = np.asarray(acq, float)
    pick = int(np.argmax(mu))
    sdm = float(sd.mean()) if sd.size else 0.0
    obs = np.array([v for _s, v in observations], float) if observations else np.zeros(1)
    return np.array([
        float(mu[pick] - mu[0]),
        float((mu[pick] - mu[0]) / (sdm + 1e-9)),
        float(mu.std()),
        sdm,
        float(sd[pick]),
        float(sd[0]),
        float(sd[pick] / (sdm + 1e-9)),
        float(acq[0] - acq[pick]),
        float(acq.std()),
        float(round_index / max(1, n_rounds)),
        float(len(obs)),
        float(obs.max()),
        float(obs.std()),
        float(len(mu)),
    ], dtype=float)


@dataclass
class LearnedGateSelector:
    """MaxMeanSelector behind a fitted gate. Abstaining returns rank 0 = plain BO."""

    model: object
    tau: float
    n_rounds: int
    acted: list[int] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.acted = []

    def __call__(self, shortlist, scores, mu, sd, observations, round_index: int = 0, **_):
        x = features(mu, sd, scores, observations, round_index, self.n_rounds)
        p = float(self.model.predict_proba(x[None, :])[0, 1])
        act = p >= self.tau
        self.acted.append(int(act))
        return int(np.argmax(np.asarray(mu, float))) if act else 0


@dataclass
class RandomGateSelector:
    """The control. Same chooser, same firing rate, no idea which rounds.

    Its rate is the gate's *offline* coverage on the held-out library, known
    before the campaign runs, so this arm is not tuned to the gated arm's
    realised behaviour.
    """

    rate: float

    def __call__(self, mu, rng=None, **_):
        r = rng if rng is not None else np.random.default_rng(0)
        act = r.random() < self.rate
        return int(np.argmax(np.asarray(mu, float))) if act else 0


def youden(score, label):
    """max_tau [P(score>=tau | right) - P(score>=tau | wrong)] and the argmax."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan"), float("inf")
    best, best_tau = -np.inf, np.inf
    for tau in np.unique(score):
        act = score >= tau
        j = act[label].mean() - act[~label].mean()
        if j > best:
            best, best_tau = float(j), float(tau)
    return best, best_tau


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--harvest-seeds", type=int, default=120,
                   help="campaigns per library used to harvest decisions")
    p.add_argument("--eval-seeds", type=int, default=200,
                   help="paired seeds for the confirming campaign arms")
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--shortlist-k", type=int, default=15)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--out", default="results/shortlist_gate.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)

    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from molbo.campaign import ArmConfig
    from molbo.clm import MaxMeanSelector, OracleShortlistSelector, RecordingSelector
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle

    n_rounds = max(1, args.budget - args.n_init)

    # ---- harvest decisions from runs that ARE plain BO -------------------
    data = {}
    for ds in args.datasets:
        name = Path(ds).stem
        oracle = LookupOracle.from_dataset(ds, value_column=args.value_column,
                                           budget=args.budget)
        lib = list(oracle.library)
        rec = RecordingSelector(dict(oracle._table), tag=name)
        run_sweep(oracle, [ArmConfig("record", strategy="bo", acquisition="ei",
                                     n_init=args.n_init, shortlist_k=args.shortlist_k,
                                     selector=rec)],
                  range(args.harvest_seeds), budget=args.budget, k=10,
                  library=lib, verbose=False)
        X, y, seeds = [], [], []
        for r in rec.records:
            mu = np.asarray(r["mu"], float)
            true = np.asarray(r["true"], float)
            if len(mu) < 2:
                continue
            X.append(features(mu, r["sd"], r["acq"], r["observations"],
                              r["round"], n_rounds))
            y.append(bool(true[int(np.argmax(mu))] >= true.max()))
            seeds.append(r["seed"])
        data[name] = {"X": np.array(X), "y": np.array(y, bool),
                      "seed": np.array(seeds), "path": ds}
        print(f"[harvest] {name:<26s}{len(y):>7d} decisions   "
              f"alpha(maxmu) = {np.mean(y):.3f}", flush=True)

    # ---- leave-one-library-out gate -------------------------------------
    print(f"\n{'=' * 78}\nLEAVE-ONE-LIBRARY-OUT GATE on MaxMeanSelector\n{'=' * 78}")
    print(f"  {'held-out library':<26s}{'alpha':>7s}{'AUROC':>8s}{'gamma':>8s}"
          f"{'shuffled':>10s}{'needed':>8s}{'verdict':>9s}")
    print("  " + "-" * 78)

    def fit(Xtr, ytr):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=3000,
                                             class_weight="balanced"))
        m.fit(Xtr, ytr)
        return m

    rng = np.random.default_rng(0)
    rows, fitted = [], {}
    for held in data:
        tr = [k for k in data if k != held]
        Xtr = np.vstack([data[k]["X"] for k in tr])
        ytr = np.concatenate([data[k]["y"] for k in tr])
        yh = data[held]["y"]
        m = fit(Xtr, ytr)
        s = m.predict_proba(data[held]["X"])[:, 1]
        a = auroc(s, yh)
        g, tau = youden(s, yh)
        # PERMUTED-LABEL CONTROL. `youden` maximises J over thresholds, which is
        # an in-sample sweep on the held-out library and so is biased upward
        # even when the gate knows nothing. The size of that bias is not
        # guessable -- it is measured, by training on shuffled labels and
        # applying the identical sweep.
        gs = []
        for _ in range(20):
            ms = fit(Xtr, rng.permutation(ytr))
            gs.append(youden(ms.predict_proba(data[held]["X"])[:, 1], yh)[0])
        g_null = float(np.mean(gs))
        # Section 13: gamma >= 0.25 took the break-even to the floor on 8 of 10
        # cells; 0.50 on the other two. The bar is 0.25 ABOVE the null, not 0.25.
        ok = (g - g_null) >= 0.25
        cov_off = float((s >= tau).mean())
        fitted[held] = (m, tau, float(a), float(g), cov_off)
        rows.append({"library": held, "alpha": float(yh.mean()), "auroc": a,
                     "gamma": g, "gamma_shuffled": g_null, "tau": float(tau),
                     "coverage_offline": cov_off, "n": int(len(yh))})
        print(f"  {held:<26s}{yh.mean():>7.3f}{a:>8.3f}{g:>+8.3f}{g_null:>+10.3f}"
              f"{0.25:>8.2f}{'CLEARS' if ok else 'short':>9s}")

    # ---- the confirming campaign ----------------------------------------
    # An offline gamma is not the result. Section 4 recorded two cases where a
    # shortlist metric inverted against campaign outcome, so the gate is run.
    print(f"\n{'=' * 78}\nCONFIRMING CAMPAIGN (the offline gamma is not the result)"
          f"\n{'=' * 78}")
    print(f"  {'library':<26s}{'maxmu':>10s}{'gated':>10s}{'rand@cov':>10s}"
          f"{'oracle':>10s}{'acts on':>9s}{'gated-rand':>12s}")
    print("  " + "-" * 88)
    for name, d in data.items():
        m, tau, a, g, cov_off = fitted[name]
        oracle = LookupOracle.from_dataset(d["path"], value_column=args.value_column,
                                           budget=args.budget)
        lib = list(oracle.library)
        sel = dict(strategy="bo", acquisition="ei", n_init=args.n_init,
                   shortlist_k=args.shortlist_k)
        arms = [ArmConfig("bo_ei", strategy="bo", acquisition="ei", n_init=args.n_init),
                ArmConfig("maxmu", selector=MaxMeanSelector(), **sel),
                ArmConfig("gated", selector=LearnedGateSelector(m, tau, n_rounds), **sel),
                ArmConfig("randgate", selector=RandomGateSelector(cov_off), **sel),
                ArmConfig("oracle", selector=OracleShortlistSelector(dict(oracle._table)),
                          **sel)]
        mt, traces = run_sweep(oracle, arms, range(args.eval_seeds),
                               budget=args.budget, k=10, library=lib, verbose=False)
        auc = {}
        for arm in arms:
            v = {x.seed: x.auc for x in mt if x.arm == arm.name}
            auc[arm.name] = np.array([v[s] for s in range(args.eval_seeds)])
        ref = auc["bo_ei"]
        cov = {a_: float(np.mean([np.mean([list(e["chosen_rank"]) != [0]
                                           for e in t.selector_log])
                                  for t in traces if t.arm == a_ and t.selector_log]))
               for a_ in ("gated", "randgate")}
        dd = auc["gated"] - auc["randgate"]
        pv = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        r = next(x for x in rows if x["library"] == name)
        r.update({"d_maxmu": float((auc["maxmu"] - ref).mean()),
                  "d_gated": float((auc["gated"] - ref).mean()),
                  "d_randgate": float((auc["randgate"] - ref).mean()),
                  "d_oracle": float((auc["oracle"] - ref).mean()),
                  "coverage": cov["gated"], "coverage_rand": cov["randgate"],
                  "gated_minus_maxmu": float((auc["gated"] - auc["maxmu"]).mean()),
                  "gated_minus_rand": float(dd.mean()), "p": float(pv)})
        print(f"  {name:<26s}{r['d_maxmu']:>+10.4f}{r['d_gated']:>+10.4f}"
              f"{r['d_randgate']:>+10.4f}{r['d_oracle']:>+10.4f}"
              f"{cov['gated']:>8.0%}{dd.mean():>+12.4f}  p={pv:.4f}")

    g_all = np.array([r["gamma"] for r in rows])
    g_null = np.array([r["gamma_shuffled"] for r in rows])
    dr = np.array([r["gated_minus_rand"] for r in rows])
    print(f"\n  mean achieved gamma          {g_all.mean():+.3f}")
    print(f"  mean gamma, labels shuffled  {g_null.mean():+.3f}   <- the bias in"
          f" the threshold sweep")
    print(f"  gamma above its own null     {(g_all - g_null).mean():+.3f}   "
          f"(section 13 bar: 0.25)")
    print(f"\n  mean gated - plain BO        {np.mean([r['d_gated'] for r in rows]):+.4f}")
    print(f"  mean gated - random gate     {dr.mean():+.4f}   <- the only column"
          f" that is evidence")
    print(f"  mean ceiling (oracle)        {np.mean([r['d_oracle'] for r in rows]):+.4f}")
    print("\n  A gate is demonstrated only if gamma clears its shuffled null by")
    print("  0.25 AND the campaign beats a random gate firing equally often.")
    print("  Offline gamma alone is the metric that inverted twice in section 4.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": args.budget, "shortlist_k": args.shortlist_k,
                               "features": FEATURES, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

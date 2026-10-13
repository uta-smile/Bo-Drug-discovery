"""Selectivity, not potency: the objective a kinase programme actually fails on.

    python scripts/selectivity_campaign.py --family jak --budget 200 --audit

Every campaign in this package so far optimises a single potency column, which
is the objective where multi-task modelling has the *weakest* architectural
case: the auxiliary targets are a source of correlated side information and
nothing more. Selectivity is different in kind. It is a **contrast between
tasks**,

    y_sel(m) = pIC50(m, t0) - max_{j != 0} pIC50(m, t_j),

and a coregionalised model can express it because it models every task; a
single-task model can only regress the composite scalar and hope. This is the
one objective where the multi-task machinery should have a structural advantage
rather than a statistical one.

It is also the objective that matters outside a benchmark. Kinase inhibitors do
not usually fail because they are insufficiently potent; they fail because they
hit something else. Off-target inhibition is a clinical failure mode, not a
metric, and a programme with a potent-but-promiscuous series has a problem that
more potency cannot solve.

**Arms.** Every arm observes the *full panel* on any compound it selects --
which is what selectivity profiling is, and what makes the comparison fair:
no arm sees more measurements than another, and the budget is counted in
compounds. Splitting a budget across targets is not re-litigated here; section
14 rejected that (0/21 targets) and at these budgets it would leave ~50
compounds per target, which tests nothing.

* ``random``   -- the floor.
* ``potency``  -- optimise on-target potency alone, score on selectivity. This
                  is what a programme does by default, and it is the baseline
                  the whole idea has to beat: if optimising potency happens to
                  deliver selectivity, none of the rest is worth doing.
* ``contrast`` -- a single-task GP regressing ``y_sel`` directly. Standard BO
                  pointed at the right objective.
* ``compose``  -- a coregionalised GP per task, composing the contrast from the
                  posterior means. The arm this script exists to test.

``compose`` uses a first-order variance for the contrast,
``sd_sel^2 ~ sd_{t0}^2 + sd_{j*}^2`` at the arg-max anti-target ``j*``, which
ignores the covariance between the two and is therefore an approximation, not a
posterior. It is stated here rather than buried because a better contrast
variance is the obvious next refinement if this arm wins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.acquisition import acquisition_scores  # noqa: E402
from molbo.audit import Audit  # noqa: E402
from molbo.metrics import auc_top_k  # noqa: E402
from molbo.surrogate import CoregionalisedGP, TanimotoGP  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_block_campaign import FAMILIES  # noqa: E402
from dense_block_h1 import build_block, drop_censored  # noqa: E402

ARMS = ("random", "potency", "contrast", "compose")


def selectivity(vals: np.ndarray, t0: int) -> np.ndarray:
    """``y(t0) - max(other targets)``, the contrast a panel is run to measure.

    Positive means the compound is more potent on the on-target than on any
    anti-target. A selective compound need not be the most potent one, which is
    the whole reason this is a different optimisation problem.
    """
    others = [j for j in range(vals.shape[1]) if j != t0]
    return vals[:, t0] - vals[:, others].max(axis=1)


def integrity(vals, t0, targets, k):
    """Is the selectivity top-k real chemistry, or the assay's reporting floor?

    This objective is a *difference*, which makes it far more exposed to
    censoring than a potency column is. If an anti-target's value is a censored
    floor -- "IC50 > 10 uM" stored as pIC50 5.00 -- then
    ``y(on) - max(off)`` is large for a reason that has nothing to do with
    selectivity, and a campaign that chases those compounds is chasing the
    limit of detection.

    ``drop_censored`` removes any compound touching a value held by >2% of a
    column, but a reporting floor sitting just under that threshold survives,
    and the contrast will find it. So the winners are inspected directly: how
    often the binding anti-target value is a *repeated* value in its own
    column, which is what a floor looks like and what a real measurement does
    not.
    """
    others = [j for j in range(vals.shape[1]) if j != t0]
    y_sel = selectivity(vals, t0)
    top = np.argsort(y_sel)[-k:]
    anti = vals[:, others]
    jstar = anti[top].argmax(axis=1)
    binding = anti[top, jstar]
    # How many compounds in the whole block share each binding value, in the
    # column it came from. A genuine measurement is near-unique.
    reps = np.array([int(np.isclose(vals[:, others[j]], row, atol=1e-4).sum())
                     for row, j in zip(binding, jstar)])
    return {
        "n_top": int(k),
        "median_shared": int(np.median(reps)),
        "max_shared": int(reps.max()),
        "frac_on_repeated": float(np.mean(reps >= 0.01 * len(vals))),
        "binding_min": float(binding.min()),
        "binding_max": float(binding.max()),
        "ontarget_median": float(np.median(vals[top, t0])),
    }


def run_campaign(K, vals, t0, *, arm, budget, batch, n_init, acquisition, rng,
                 audit):
    """One selectivity campaign. Returns the selected indices, in order.

    Selecting a compound reveals the whole panel row for it -- every target --
    because that is how a selectivity assay is run. So every arm holds the same
    information about every compound it has paid for, and they differ only in
    what they do with it.
    """
    n_pool = len(vals)
    T = vals.shape[1]
    others = [j for j in range(T) if j != t0]
    y_sel = selectivity(vals, t0)

    observed = list(rng.choice(n_pool, size=n_init, replace=False))
    while len(observed) < budget:
        obs = np.asarray(observed, dtype=np.intp)
        cand = np.setdiff1d(np.arange(n_pool), obs)
        take = min(batch, budget - len(observed))

        if arm == "random":
            observed.extend(rng.choice(cand, size=take, replace=False).tolist())
            continue

        if arm in ("potency", "contrast"):
            # The two differ only in which scalar is regressed, which is the
            # comparison this script exists to make: `potency` is blind to the
            # anti-targets although it has paid for them.
            target = vals[:, t0] if arm == "potency" else y_sel
            gp = TanimotoGP().fit_gram(K[np.ix_(obs, obs)], target[obs])
            mu, sd = gp.predict_gram(K[np.ix_(cand, obs)],
                                     k_diag=np.ones(len(cand)), return_std=True)
            best = float(target[obs].max())
        else:
            # One coregionalised fit per task, that task primary and the rest
            # auxiliary, all on the same observed compounds (a blocked design --
            # the panel). Then compose.
            fits = []
            for j in range(T):
                aux = [k for k in range(T) if k != j]
                model = CoregionalisedGP()
                model.fit_gram(K, [obs] * T, [vals[obs, j]] + [vals[obs, a]
                                                               for a in aux])
                fits.append(model.predict_gram(K, cand, return_std=True))
            mu_all = np.stack([f[0] for f in fits])      # (T, n_cand)
            sd_all = np.stack([f[1] for f in fits])
            anti, anti_sd = mu_all[others], sd_all[others]
            rows = np.arange(len(cand))
            jstar = anti.argmax(axis=0)
            mu = mu_all[t0] - anti[jstar, rows]
            sd = np.hypot(sd_all[t0], anti_sd[jstar, rows])
            best = float(y_sel[obs].max())

        scores = acquisition_scores(acquisition, mu, sd, best=best, rng=rng)
        order = np.argsort(-(scores + 1e-12 * rng.standard_normal(len(scores))))
        observed.extend(cand[order[:take]].tolist())

    idx = np.asarray(observed, dtype=np.intp)
    audit.cell()
    audit.exactly(idx, budget, f"{arm} t0={t0} selections")
    return idx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="jak", choices=sorted(FAMILIES))
    p.add_argument("--budget", type=int, default=200,
                   help="compounds assayed; each reveals the whole panel row")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--acquisition", default="ei")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--out", default="results/selectivity.json")
    args = p.parse_args(argv)

    from molbo.surrogate import fingerprint_matrix, tanimoto_gram

    t0_ = time.time()
    mapping = FAMILIES[args.family]
    block, _ = build_block(mapping)
    targets = [t for t in mapping.values() if t in block.columns]
    if not args.keep_censored:
        block, _ = drop_censored(block, targets)
    if len(block) < args.budget + args.k:
        raise SystemExit(f"only {len(block)} compounds survive the filters; "
                         f"budget {args.budget} + top-{args.k} needs more")

    X = fingerprint_matrix(block.smiles.tolist())
    K = tanimoto_gram(X, X)
    vals = block[targets].to_numpy(float)
    print(f"{args.family}: {len(block):,} compounds x {len(targets)} targets, "
          f"budget {args.budget} compounds (full panel each), "
          f"batch {args.batch} ({time.time() - t0_:.0f}s)")

    audit = Audit(enabled=args.audit)
    arms = tuple(args.arms)
    acc = {a: {"recall": [], "auc": []} for a in arms}
    per_target: dict = {}

    for ti, t in enumerate(targets):
        y_sel = selectivity(vals, ti)
        true_top = set(np.argsort(y_sel)[-args.k:].tolist())
        # How much of the selectivity top-k is also potency top-k. If it were
        # most of them, `potency` would win for free and the objective would not
        # be a separate problem.
        pot_top = set(np.argsort(vals[:, ti])[-args.k:].tolist())
        share = len(true_top & pot_top) / args.k
        integ = integrity(vals, ti, targets, args.k)
        flag = " <-- CENSORING SUSPECT" if integ["frac_on_repeated"] >= 0.5 else ""
        print(f"    {t:<8s} selectivity range {y_sel.min():+.2f}..{y_sel.max():+.2f}"
              f"   top-{args.k} overlap with potency top-{args.k}: {share:.0%}",
              flush=True)
        print(f"             top-{args.k} binding anti-target value "
              f"{integ['binding_min']:.2f}..{integ['binding_max']:.2f}; "
              f"shared by {integ['median_shared']} compounds (median), "
              f"{integ['max_shared']} (max); "
              f"{integ['frac_on_repeated']:.0%} sit on a repeated value{flag}",
              flush=True)
        per_target[t] = {"potency_overlap": share, "integrity": integ}

        for r in range(args.repeats):
            for arm in arms:
                idx = run_campaign(
                    K, vals, ti, arm=arm, budget=args.budget, batch=args.batch,
                    n_init=args.n_init, acquisition=args.acquisition,
                    rng=np.random.default_rng(args.seed + 1009 * r + 97 * ti),
                    audit=audit)
                acc[arm]["recall"].append(
                    len(true_top & set(idx.tolist())) / args.k)
                acc[arm]["auc"].append(auc_top_k(y_sel[idx].tolist(), k=args.k))
        for arm in arms:
            lo = ti * args.repeats
            per_target[t][arm] = float(np.mean(acc[arm]["recall"][lo:]))

    report = {"family": args.family, "budget": args.budget, "batch": args.batch,
              "k": args.k, "repeats": args.repeats, "n_pool": int(len(block)),
              "targets": targets, "objective": "pIC50(on) - max(off)",
              "per_target": per_target, "arms": {}}

    print(f"\n    {'arm':<12s}{'selectivity top-' + str(args.k):>22s}{'vs potency':>14s}")
    print("    " + "-" * 48)
    base = np.asarray(acc["potency"]["recall"]) if "potency" in arms else None
    for arm in arms:
        rec = np.asarray(acc[arm]["recall"])
        delta = ("" if base is None or arm == "potency"
                 else f"{(rec - base).mean():>+12.1%}")
        print(f"    {arm:<12s}{rec.mean():>22.1%}{delta:>14s}")
        report["arms"][arm] = {
            "recall_mean": float(rec.mean()),
            "recall_sem": float(rec.std() / np.sqrt(len(rec))),
            "auc_mean": float(np.mean(acc[arm]["auc"])),
            "recall": [float(x) for x in rec],
        }

    if "compose" in arms and "contrast" in arms:
        d = (np.asarray(acc["compose"]["recall"])
             - np.asarray(acc["contrast"]["recall"]))
        print(f"\n    compose - contrast: {d.mean():+.1%} "
              f"+-{d.std() / np.sqrt(len(d)):.1%}  "
              f"({(d > 0).mean():.0%} of campaigns)")
        print("    (does modelling the panel and composing beat regressing the "
              "contrast?)")
        report["compose_minus_contrast"] = float(d.mean())

    print(f"\n    per-target selectivity recall")
    print(f"    {'on-target':<10s}" + "".join(f"{a:>11s}" for a in arms)
          + f"{'pot.overlap':>13s}")
    for t in targets:
        print(f"    {t:<10s}" + "".join(f"{per_target[t][a]:>11.1%}" for a in arms)
              + f"{per_target[t]['potency_overlap']:>13.0%}")

    audit.report("[selectivity]", clean_lines=(
        "every arm spent exactly the budget, with no compound selected twice",))
    report["audit"] = audit.to_json()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0_:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

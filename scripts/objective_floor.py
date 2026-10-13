"""How much of a DOCKSTRING objective is free, as a function of how it is built?

    python scripts/objective_floor.py --out results/objective_floor.json

FINDINGS section 14 found that the published objectives are ~93% a descriptor
that costs nothing: ranking ``score_jak2`` by QED alone, with zero assays,
recovers about half of its top-50.  That closes those objectives for H2 and H3,
because an LLM warm start or an adaptive controller would be competing for the
2.3 points that 80 assays actually buy.

The fix is a dial, ``- w (1 - QED)``, and the temptation is to pick a ``w`` that
looks reasonable.  This measures it instead.  For each treatment of
drug-likeness it reports:

* **floor** -- top-k recall of the best *free* ranking, which is QED for a
  weighted objective and pure chance for a filtered one.  This is what an
  optimiser scores before it has measured anything.
* **reachable** -- top-k recall of a Tanimoto GP given the budget, which is what
  the optimiser can get to.
* **headroom** -- reachable minus floor.  The band every later claim about warm
  starts, controllers or acquisition has to fit inside.  When it is 2 points
  wide, nothing fits inside it.
* **learnable** -- Spearman of the GP against the objective, to catch the other
  failure: an objective with no free component and no signal either is not an
  improvement, it is noise.

Treatments compared:

* ``w = 10`` -- the published objective, Kubincova & Mobley's.
* ``w = 5, 2, 1`` -- the same thing with the penalty shrunk.
* ``w = 0, QED >= 0.5`` -- drug-likeness as an admission filter on the library
  instead of a term in the score.  The molecules that fail are *removed*, not
  scored badly, so there is nothing free left to rank by.

Pick the treatment from the table, not from taste.  Docking scores, not
measurements: see ``molbo/dockstring.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from molbo.metrics import spearman0 as _rho  # noqa: E402

from molbo import dockstring as ds  # noqa: E402
from molbo.surrogate import TanimotoGP, fingerprint_matrix, tanimoto_gram  # noqa: E402

W_GRID: tuple[float, ...] = (10.0, 5.0, 2.0, 1.0, 0.0)
"""Drug-likeness weights swept, published value first."""

N_GRID = (10, 30, 80)
"""Budgets. The same grid as ``transfer_headroom.py``, trimmed: the floor does
not move with n and only the reachable column needs more than three points."""




def _recall(truth: np.ndarray, pred: np.ndarray, k: int) -> float:
    top = set(np.argsort(truth)[-k:].tolist())
    return len(top & set(np.argsort(pred)[-k:].tolist())) / k


def _chance(n_rows: int, k: int) -> float:
    """Recall a uniformly random ranking earns: k of n rows, in expectation."""
    return k / n_rows


def evaluate(sub, K, obj, *, topk, n_repeats, seed):
    """One treatment: its free floor, and what a GP reaches above it."""
    admit = obj.admits(sub)
    idx = np.flatnonzero(admit)
    if len(idx) < 50:
        raise SystemExit(f"{obj.name} admits only {len(idx)} molecules")
    # A top-k as wide as the pool is not a top-k: every arm returns every row
    # and the column becomes a function of k and the pool, not of any method.
    if topk >= len(idx) - max(N_GRID):
        raise SystemExit(
            f"--topk {topk} leaves no room: {obj.name} admits {len(idx)} "
            f"molecules and spends up to {max(N_GRID)} on training. "
            f"Use --topk below {len(idx) - max(N_GRID)} or a larger --pool.")
    y = np.asarray(obj(sub), dtype=float)[idx]
    free = np.asarray(obj.free(sub), dtype=float)[idx]
    Ki = K[np.ix_(idx, idx)]
    n_rows = len(idx)

    # The free ranking. With w = 0 the free term is constant, so a free ranking
    # has no information and chance is the honest floor; reporting recall of a
    # constant ordering would report an artefact of argsort's tie-breaking.
    if float(np.ptp(free)) > 0:
        floor = _recall(y, free, topk)
        floor_kind = "QED"
    else:
        floor = _chance(n_rows, topk)
        floor_kind = "chance"
    floor_rho = _rho(y, free) if float(np.ptp(free)) > 0 else 0.0

    assayed = np.asarray(obj.assayed(sub), dtype=float)[idx]

    rows = {}
    for n in N_GRID:
        rec_j, rec_i, rho = [], [], []
        for r in range(n_repeats):
            g = np.random.default_rng(seed + 101 * r + n)
            tr = g.choice(n_rows, size=n, replace=False)
            te = np.setdiff1d(np.arange(n_rows), tr)
            Ktr, Kte = Ki[np.ix_(tr, tr)], Ki[np.ix_(te, tr)]

            # Two arms. `naive` regresses the finished objective, which means
            # relearning the free term from fingerprints. `informed` regresses
            # only the assayed part and adds the free term back -- what a
            # practitioner who has read the objective would do, and the arm the
            # floor has to be compared against for the comparison to be fair.
            naive = TanimotoGP().fit_gram(Ktr, y[tr]).predict_gram(
                Kte, return_std=False)
            gi = TanimotoGP().fit_gram(Ktr, assayed[tr])
            informed = gi.predict_gram(Kte, return_std=False) + free[te]

            # Score on the same rows the floor is scored on: the floor knows the
            # training rows too, so hiding them from one side only would flatter
            # whichever arm had the smaller training set.
            for pred, acc in ((naive, rec_j), (informed, rec_i)):
                whole = np.empty(n_rows)
                whole[te] = pred
                whole[tr] = y[tr]
                acc.append(_recall(y, whole, topk))
            rho.append(_rho(y[te], informed))

        best_arm = max(float(np.mean(rec_j)), float(np.mean(rec_i)))
        rows[n] = {"recall": best_arm,
                   "recall_naive": float(np.mean(rec_j)),
                   "recall_informed": float(np.mean(rec_i)),
                   "recall_sem": float(np.std(rec_i) / np.sqrt(n_repeats)),
                   "learnable_rho": float(np.mean(rho)),
                   "headroom": best_arm - floor}
    return {"name": obj.name, "qed_weight": obj.qed_weight, "qed_min": obj.qed_min,
            "n_admitted": n_rows, "floor": floor, "floor_kind": floor_kind,
            "floor_rho": floor_rho, "n": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(ds.DATASET))
    p.add_argument("--base", default="jak2", choices=("jak2", "jak2_only", "ppar"),
                   help="which published objective's assayed terms to sweep")
    p.add_argument("--pool", type=int, default=2000)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--qed-min", type=float, default=ds.QED_MIN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/objective_floor.json")
    args = p.parse_args(argv)

    t0 = time.time()
    print(f"loading {args.dataset} ...", flush=True)
    df = ds.load(args.dataset)
    cols = list(ds.TARGETS)
    full = df[np.isfinite(df[cols].to_numpy(dtype=float)).all(axis=1)]
    full = full[np.isfinite(full.QED.to_numpy(dtype=float))].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    sub = full.iloc[rng.choice(len(full), size=args.pool, replace=False)]
    sub = sub.reset_index(drop=True)
    X = fingerprint_matrix(sub.smiles.tolist())
    K = tanimoto_gram(X, X)
    print(f"pool {len(sub):,}, fingerprints + gram, {time.time() - t0:.0f}s")

    base = ds.OBJECTIVES[args.base]
    treatments = [base.reweighted(w) for w in W_GRID]
    treatments.append(base.reweighted(0.0, args.qed_min))

    print(f"\nobjective={args.base}  pool={len(sub)}  top-{args.topk}  "
          f"{args.repeats} repeats")
    print(f"\n    {'treatment':<22s}{'admits':>8s}{'floor':>9s}{'via':>8s}"
          + "".join(f"{'n=' + str(n) + ' recall':>12s}{'hdrm':>8s}" for n in N_GRID))
    print("    " + "-" * (47 + 20 * len(N_GRID)))

    out = []
    for obj in treatments:
        rec = evaluate(sub, K, obj, topk=args.topk, n_repeats=args.repeats,
                       seed=args.seed)
        out.append(rec)
        label = (f"w={obj.qed_weight:g}" if obj.qed_min is None
                 else f"w=0, QED>={obj.qed_min:g}")
        cells = "".join(
            f"{rec['n'][n]['recall']:>12.1%}{rec['n'][n]['headroom']:>+8.1%}"
            for n in N_GRID)
        print(f"    {label:<22s}{rec['n_admitted']:>8d}{rec['floor']:>9.1%}"
              f"{rec['floor_kind']:>8s}{cells}", flush=True)

    print(f"\n    free share -- how much of the best arm's recall it already had "
          f"before assaying")
    print(f"    {'treatment':<22s}" + "".join(f"{'n=' + str(n):>10s}" for n in N_GRID)
          + f"{'learnable@' + str(N_GRID[-1]):>16s}")
    for rec in out:
        label = (f"w={rec['qed_weight']:g}" if rec["qed_min"] is None
                 else f"w=0, QED>={rec['qed_min']:g}")
        shares = []
        for n in N_GRID:
            r = rec["n"][n]["recall"]
            s = rec["floor"] / r if r > 0 else float("nan")
            rec["n"][n]["free_share"] = float(s)
            shares.append(s)
        print(f"    {label:<22s}" + "".join(f"{s:>10.0%}" for s in shares)
              + f"{rec['n'][N_GRID[-1]]['learnable_rho']:>16.3f}")

    # Verdict: the usable treatment is the one with the widest band between what
    # is free and what is reachable, provided the objective is still learnable.
    def _band(rec):
        return max(v["headroom"] for v in rec["n"].values())

    def _band_sem(rec):
        n = max(rec["n"], key=lambda k: rec["n"][k]["headroom"])
        return rec["n"][n]["recall_sem"]

    if not any(np.isfinite(_band(r)) for r in out):
        print("\n    verdict:")
        print("      no treatment produced a finite band -- every arm failed to fit.")
        print("      Not a result about the objectives; check --pool and --topk.")
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"base": args.base, "treatments": out,
                                    "verdict": {"status": "no_fits"}}, indent=2),
                        encoding="utf-8")
        return 1

    best = max(out, key=_band)
    # Prefer the filtered treatment when it is not measurably worse. A weight is
    # a knob fitted to this benchmark and would have to be re-justified on the
    # next one; an admission filter is a statement about which molecules the
    # question is asked over, and survives the move.
    filtered = next((r for r in out if r["qed_min"] is not None), None)
    if (filtered is not None and filtered is not best
            and _band(best) - _band(filtered) <= 3 * _band_sem(best)):
        print(f"\n    (w={best['qed_weight']:g} bands {_band(best):+.1%} and the filter "
              f"{_band(filtered):+.1%} -- within noise;")
        print("     taking the filter, which is a choice of library rather than a")
        print("     weight fitted to this benchmark.)")
        best = filtered
    published = out[0]
    pub_h = _band(published)
    best_h = _band(best)
    best_label = (f"w={best['qed_weight']:g}" if best["qed_min"] is None
                  else f"w=0 with QED >= {best['qed_min']:g}")
    print("\n    verdict:")
    print(f"      published (w=10): floor {published['floor']:.1%}, "
          f"headroom {pub_h:+.1%}")
    print(f"      widest band:      {best_label} -- floor {best['floor']:.1%}, "
          f"headroom {best_h:+.1%}")
    n_last = N_GRID[-1]
    pub_share = published["n"][n_last]["free_share"]
    best_share = best["n"][n_last]["free_share"]
    print(f"      free share at n={n_last}: published {pub_share:.0%}, "
          f"{best_label} {best_share:.0%}")
    if pub_h <= 0:
        print("      The published objective's best arm does not reach its own free")
        print(f"      floor: {pub_h:+.1%}. Ranking by QED beats measuring.")
    if best_share < 0.5 <= pub_share:
        print(f"      The published objective hands over {pub_share:.0%} of the answer "
              "before any")
        print("      assay, so an H2 or H3 effect would have to show up inside the "
              f"{pub_h:+.1%}")
        print(f"      that is left. On {best_label} the arm earns "
              f"{1 - best_share:.0%} of its own")
        print("      recall. Run H2 and H3 there; quote nothing on the published")
        print("      objective without its floor beside it.")
    elif best_h > pub_h + 3 * _band_sem(best):
        print(f"      Wider band ({best_h:+.1%} vs {pub_h:+.1%}), same free share. "
              "Worth moving to,")
        print("      but the gain is room to measure in, not a cleaner objective.")
    else:
        print("      No treatment opens a materially wider band. The objective's")
        print("      difficulty is not in its QED term, and a different assayed")
        print("      formulation -- not a different weight -- is what is needed.")
    verdict = {"published_headroom": pub_h, "best_headroom": best_h,
               "published_free_share": pub_share, "best_free_share": best_share,
               "best_treatment": best_label}

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"base": args.base, "pool": len(sub),
                                "topk": args.topk, "repeats": args.repeats,
                                "seed": args.seed, "treatments": out,
                                "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

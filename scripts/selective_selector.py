"""The same question on the hook where abstaining is free: BO's shortlist.

    python scripts/selective_selector.py --datasets data/chembl/cruzipain.csv ...

``scripts/selective_headroom.py`` asks whether a chooser that may decline beats
one that must act, on the *region* hook. There, declining costs a free bandit's
exploration and the ceiling itself is near zero on a random target (FINDINGS
section 3), so the answer is bounded from above by something small.

The shortlist hook is the opposite shape, and it is the one place in this
project where a gate is structurally cheap:

* **Abstaining is exactly plain BO.** The chooser picks a position in BO's
  EI-ranked top-k; position 0 is what plain BO would have bought. So a round it
  declines costs nothing at all, not merely "less".
* **The ceiling is live.** Section 1: +0.031 AUC (dz 0.89), and the best free
  control takes **0%** of it -- unlike the region hook, where a ten-line bandit
  takes 40-50%.
* **The decision is per round, not per campaign**, so a 30-call campaign asks
  it twenty times. Accuracy and calibration are both measurable in one run.

What is swept is a chooser defined by two numbers and nothing else:

``alpha``  P(it names the shortlist's truly best compound)
``gamma``  P(act | right) - P(act | wrong), with P(act | right) held at 1.

``gamma`` is exactly Youden's J for the gate, and a binary decision's AUROC
against correctness is ``(1 + gamma) / 2`` -- so gamma = 0 is a chooser with no
idea when it is right (every chooser in this project), gamma = 1 is one that
knows exactly. The alpha axis alone reproduces the usual break-even question;
the gamma axis is the one the Jev-style framing adds, and the point of the grid
is to read off how much of one buys how much of the other.

The gamma = 0 column is therefore the committing chooser -- it acts on every
round, right or wrong -- and it is the only column this project has ever
measured. Coverage falls as gamma rises, because a higher gamma means more of
the wrong picks are declined. The cell to read first is bottom-left to
top-right: **alpha low, gamma high** is a chooser that is usually wrong and
knows it, and whether that is worth anything is the entire question.
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


@dataclass
class SyntheticGatedSelector:
    """A chooser with a dialled accuracy and a dialled knowledge of it.

    ``values`` is the lookup table and is used only to *define* what a correct
    pick is -- the chooser is a coin flip, not a model, so no information from
    it reaches the campaign except through ``accuracy`` and ``gate_quality``.

    Abstaining returns position 0, which is BO's own acquisition argmax. That
    is the whole reason this hook is worth gating: a declined round is not a
    cheaper mistake, it is not a mistake.
    """

    values: dict[str, float]
    accuracy: float = 1.0
    gate_quality: float = 0.0
    """gamma = P(act | right) - P(act | wrong), with P(act | right) = 1."""
    log: list[dict] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self.log = []

    def __call__(self, shortlist, rng=None, round_index: int = 0, **_):
        r = rng if rng is not None else np.random.default_rng(0)
        n = len(shortlist)
        if n == 1:
            return 0
        true = np.array([self.values[s] for s in shortlist], float)
        best = int(np.argmax(true))
        if r.random() < self.accuracy:
            pick, right = best, True
        else:
            others = [i for i in range(n) if i != best]
            pick, right = int(others[r.integers(len(others))]), False
        p_act = 1.0 if right else max(0.0, 1.0 - self.gate_quality)
        acted = bool(r.random() < p_act)
        self.log.append({"round": int(round_index), "right": right,
                         "acted": acted, "pick": pick})
        return pick if acted else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[30, 60])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--shortlist-k", type=int, default=15,
                   help="15 matches the section-1 ceiling measurement")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="default: 1/shortlist_k (which *is* the free random "
                        "control, and so is the bar the grid is read against) "
                        "then 0.2 0.4 0.6 0.8 1.0")
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--out", default="results/selective_selector.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)
    # alpha = 1/k with gamma = 0 is a uniform draw from the shortlist, i.e.
    # exactly RandomShortlistSelector. Putting it on the grid makes the free
    # control a *cell*, so the break-even is read against it rather than
    # against plain BO -- any deviation from the EI argmax adds exploration,
    # and at these budgets exploration alone can win.
    a0 = round(1.0 / args.shortlist_k, 4)
    if args.alphas is None:
        args.alphas = [a0, 0.2, 0.4, 0.6, 0.8, 1.0]

    from scipy.stats import wilcoxon

    from molbo.campaign import ArmConfig
    from molbo.clm import MaxMeanSelector, OracleShortlistSelector
    from molbo.experiment import run_sweep
    from molbo.llm import RandomShortlistSelector
    from molbo.oracles import LookupOracle

    rows = []
    for ds in args.datasets:
        oracle = LookupOracle.from_dataset(ds, value_column=args.value_column,
                                           budget=max(args.budgets))
        lib = list(oracle.library)
        values = dict(oracle._table)
        name = Path(ds).stem
        print(f"\n{'=' * 74}\n{name}  ({len(lib)} compounds, shortlist "
              f"{args.shortlist_k}, {args.seeds} seeds)\n{'=' * 74}", flush=True)

        for budget in args.budgets:
            oracle.budget = budget
            arms = [ArmConfig("bo_ei", strategy="bo", acquisition="ei",
                              n_init=args.n_init)]
            sel = dict(strategy="bo", acquisition="ei", n_init=args.n_init,
                       shortlist_k=args.shortlist_k)
            arms += [
                ArmConfig("sel_random", selector=RandomShortlistSelector(), **sel),
                ArmConfig("sel_maxmu", selector=MaxMeanSelector(), **sel),
                ArmConfig("sel_oracle", selector=OracleShortlistSelector(values), **sel),
            ]
            grid = []
            for a in args.alphas:
                for g in args.gammas:
                    nm = f"g_a{int(round(100 * a)):03d}_g{int(round(100 * g)):03d}"
                    grid.append((nm, a, g))
                    arms.append(ArmConfig(
                        nm, selector=SyntheticGatedSelector(values, a, g), **sel))
            m, _ = run_sweep(oracle, arms, range(args.seeds), budget=budget, k=10,
                             library=lib, verbose=False)
            auc = {}
            for a in arms:
                v = {x.seed: x.auc for x in m if x.arm == a.name}
                auc[a.name] = np.array([v[s] for s in range(args.seeds)])
            ref = auc["bo_ei"]

            def d(nm):
                x = auc[nm] - ref
                p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                return float(x.mean()), float(p)

            print(f"\n  budget {budget}")
            for nm in ("sel_random", "sel_maxmu", "sel_oracle"):
                dd, pp = d(nm)
                print(f"    {nm:<12s}{dd:>+9.4f}  p={pp:.4f}")
                rows.append({"dataset": name, "budget": budget, "arm": nm,
                             "alpha": None, "gamma": None, "dauc": dd, "p": pp})

            print(f"\n    dAUC vs plain BO      gamma = P(act|right) - P(act|wrong)")
            print("    alpha |" + "".join(f"{g:>10.2f}" for g in args.gammas))
            print("    ------+" + "-" * (10 * len(args.gammas)))
            for a in args.alphas:
                cells = []
                for g in args.gammas:
                    nm = f"g_a{int(round(100 * a)):03d}_g{int(round(100 * g)):03d}"
                    dd, pp = d(nm)
                    cells.append(dd)
                    rows.append({"dataset": name, "budget": budget, "arm": nm,
                                 "alpha": a, "gamma": g, "dauc": dd, "p": pp})
                print(f"    {a:>5.2f} |" + "".join(f"{c:>+10.4f}" for c in cells))

            cell = lambda a, g: next(
                r["dauc"] for r in rows if r["dataset"] == name
                and r["budget"] == budget and r["alpha"] == a and r["gamma"] == g)
            free = cell(a0, 0.0)
            print(f"\n    free control (alpha={a0:.3f}, gamma=0, i.e. a uniform "
                  f"draw) {free:+.4f}")
            print("    break-even accuracy (lowest alpha beating it), per gamma")
            for g in args.gammas:
                ok = [a for a in args.alphas if cell(a, g) > free]
                print(f"      gamma {g:.2f}  ->  "
                      f"{f'{min(ok):.0%}' if ok else '>100%'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": args.seeds, "shortlist_k": args.shortlist_k,
                               "alphas": args.alphas, "gammas": args.gammas,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # ---- pooled view -----------------------------------------------------
    print(f"\n\n{'=' * 74}\nPOOLED over "
          f"{len({r['dataset'] for r in rows})} libraries x "
          f"{len(args.budgets)} budgets\n{'=' * 74}")
    print("    alpha |" + "".join(f"{g:>10.2f}" for g in args.gammas))
    print("    ------+" + "-" * (10 * len(args.gammas)))
    for a in args.alphas:
        cells = [np.mean([r["dauc"] for r in rows
                          if r["alpha"] == a and r["gamma"] == g])
                 for g in args.gammas]
        print(f"    {a:>5.2f} |" + "".join(f"{c:>+10.4f}" for c in cells))
    pooled = lambda a, g: float(np.mean([r["dauc"] for r in rows
                                         if r["alpha"] == a and r["gamma"] == g]))
    free = pooled(a0, 0.0)
    print(f"\n    free control (alpha={a0:.3f}, gamma=0) {free:+.4f}")
    print("    break-even accuracy against it, pooled")
    for g in args.gammas:
        ok = [a for a in args.alphas if pooled(a, g) > free]
        print(f"      gamma {g:.2f}  ->  {f'{min(ok):.0%}' if ok else '>100%'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

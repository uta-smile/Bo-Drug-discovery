"""Can a *free* gate rescue a real selector that is known to lose?

    python scripts/gated_selector.py --datasets data/chembl/cruzipain.csv ...

``scripts/selective_selector.py`` sweeps a synthetic chooser and says how much
knowing-when-you-are-right is worth. This asks the other half: does any signal
available at run time, with no answer key and no training, actually carry that
knowledge?

The subject is ``MaxMeanSelector`` -- buy the shortlist's highest posterior
mean. It is the sharpest case in the project because it is *good at the wrong
thing*: 3.4x plain BO's offline gain at picking the most potent candidate, and
campaign AUC -0.0195 (p=0.011) with top-1% recall halved (FINDINGS section 4).
The diagnosis there was mechanical -- buying the posterior argmax never spends a
call resolving uncertainty -- and a mechanical diagnosis implies a gate: there
are rounds where exploiting is the right call and rounds where it is not, and
they are distinguishable *before* the answer is known.

Gates, all computable from what the selector is already handed:

``late(f)``     act only in the last ``1-f`` of rounds. The GP is best late, so
                if the diagnosis is right this is where exploitation is safe.
``early(f)``    the same gate reversed. It is here because a gate that helps
                must beat its own mirror image, or the gain was the reduced
                *amount* of exploitation, not its placement.
``margin(t)``   act only when the shortlist's best posterior mean beats rank 0's
                by at least ``t`` posterior standard deviations -- the closest
                thing here to a confidence score on the selector's own pick.
``settled(t)``  act only once the shortlist's mean posterior sd has fallen to
                ``t`` of what it was at the first decision: exploit when the
                surrogate has stopped learning quickly.
``rand(c)``     act on a random ``c`` of rounds. **The control that decides the
                section.** Every gate above reduces how often the selector acts,
                and merely acting less is worth something on its own; only the
                gap between a gate and ``rand`` at the same coverage is evidence
                that the gate knows *which* rounds.

The same shape as the scaffold-hopping result in section 11: directed beats
random, and the gap is the finding.
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
class GatedSelector:
    """Wrap a selector in an abstention rule. Abstaining returns rank 0, which
    is BO's own acquisition argmax -- so a declined round is plain BO exactly."""

    rule: str
    param: float
    n_rounds: int = 20
    coverage: list[int] = field(default_factory=list, init=False)
    _sd0: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.coverage = []
        self._sd0 = None

    def _act(self, mu, sd, round_index, rng) -> bool:
        mu = np.asarray(mu, float)
        sd = np.asarray(sd, float)
        msd = float(np.mean(sd)) if sd.size else 0.0
        if self._sd0 is None and msd > 0:
            self._sd0 = msd
        f = round_index / max(1, self.n_rounds)
        if self.rule == "late":
            return f >= self.param
        if self.rule == "early":
            return f < self.param
        if self.rule == "margin":
            ref = msd if msd > 0 else 1.0
            return bool((mu.max() - mu[0]) >= self.param * ref)
        if self.rule == "settled":
            return bool(self._sd0 and msd <= self.param * self._sd0)
        if self.rule == "rand":
            return bool((rng or np.random.default_rng(0)).random() < self.param)
        if self.rule == "always":
            return True
        raise ValueError(self.rule)

    def __call__(self, shortlist, mu=None, sd=None, rng=None, round_index: int = 0, **_):
        mu = np.zeros(len(shortlist)) if mu is None else np.asarray(mu, float)
        sd = np.zeros(len(shortlist)) if sd is None else np.asarray(sd, float)
        act = self._act(mu, sd, round_index, rng)
        self.coverage.append(int(act))
        return int(np.argmax(mu)) if act else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[30, 60])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--shortlist-k", type=int, default=15)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--out", default="results/gated_selector.json")
    return p.parse_args(argv)


GATES = [("always", 0.0), ("late", 0.25), ("late", 0.5), ("late", 0.75),
         ("early", 0.25), ("early", 0.5), ("early", 0.75),
         ("margin", 0.25), ("margin", 0.5), ("margin", 1.0),
         ("settled", 0.98), ("settled", 0.95), ("settled", 0.9),
         ("rand", 0.25), ("rand", 0.5), ("rand", 0.75)]


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    args = parse_args(argv)

    from scipy.stats import wilcoxon

    from molbo.campaign import ArmConfig
    from molbo.clm import OracleShortlistSelector
    from molbo.experiment import run_sweep
    from molbo.oracles import LookupOracle

    rows = []
    for ds in args.datasets:
        oracle = LookupOracle.from_dataset(ds, value_column=args.value_column,
                                           budget=max(args.budgets))
        lib = list(oracle.library)
        values = dict(oracle._table)
        name = Path(ds).stem
        print(f"\n{'=' * 78}\n{name}  ({len(lib)} compounds, shortlist "
              f"{args.shortlist_k}, {args.seeds} seeds)\n{'=' * 78}", flush=True)

        for budget in args.budgets:
            oracle.budget = budget
            n_rounds = max(1, budget - args.n_init)
            sel = dict(strategy="bo", acquisition="ei", n_init=args.n_init,
                       shortlist_k=args.shortlist_k)
            arms = [ArmConfig("bo_ei", strategy="bo", acquisition="ei",
                              n_init=args.n_init),
                    ArmConfig("sel_oracle", selector=OracleShortlistSelector(values),
                              **sel)]
            made = []
            for rule, prm in GATES:
                nm = f"maxmu_{rule}{prm:g}".replace(".", "")
                made.append((nm, rule, prm))
                arms.append(ArmConfig(nm, selector=GatedSelector(rule, prm, n_rounds),
                                      **sel))
            m, traces = run_sweep(oracle, arms, range(args.seeds), budget=budget,
                                  k=10, library=lib, verbose=False)
            auc = {}
            for a in arms:
                v = {x.seed: x.auc for x in m if x.arm == a.name}
                auc[a.name] = np.array([v[s] for s in range(args.seeds)])
            # How often the arm actually left rank 0, read off the campaign's
            # own log. Not the nominal gate rate: a gate can fire on a round
            # where the posterior argmax *is* rank 0, and that round is plain
            # BO whatever the gate said. Matching the random control on the
            # measured rate is what makes the comparison about placement.
            cov = {}
            for t in traces:
                if t.selector_log:
                    cov.setdefault(t.arm, []).append(
                        float(np.mean([list(e["chosen_rank"]) != [0]
                                       for e in t.selector_log])))
            ref = auc["bo_ei"]

            def d(nm):
                x = auc[nm] - ref
                p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                return float(x.mean()), float(p)

            print(f"\n  budget {budget} ({n_rounds} decisions per run)")
            dd, pp = d("sel_oracle")
            print(f"    {'sel_oracle (ceiling)':<24s}{dd:>+9.4f}  p={pp:.4f}")
            rows.append({"dataset": name, "budget": budget, "arm": "sel_oracle",
                         "rule": "oracle", "param": None, "dauc": dd, "p": pp,
                         "coverage": 1.0})
            hdr = (f"    {'gate':<24s}{'acts on':>9s}{'dAUC':>10s}{'p':>9s}"
                   f"{'vs rand@cov':>13s}")
            print(hdr + "\n    " + "-" * (len(hdr) - 4))
            rand = [(float(np.mean(cov.get(nm, [1.0]))), d(nm)[0])
                    for nm in ("maxmu_rand025", "maxmu_rand05", "maxmu_rand075",
                               "maxmu_always0")]
            for nm, rule, prm in made:
                dd, pp = d(nm)
                c = float(np.mean(cov.get(nm, [1.0])))
                gap = dd - min(rand, key=lambda x: abs(x[0] - c))[1]
                rows.append({"dataset": name, "budget": budget, "arm": nm,
                             "rule": rule, "param": prm, "dauc": dd, "p": pp,
                             "coverage": c, "vs_rand_at_coverage": gap})
                tag = "" if rule == "rand" else f"{gap:>+13.4f}"
                print(f"    {nm:<24s}{c:>8.0%}{dd:>+10.4f}{pp:>9.4f}{tag:>13s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": args.seeds, "shortlist_k": args.shortlist_k,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print(f"\n\n{'=' * 78}\nPOOLED over "
          f"{len({r['dataset'] for r in rows})} libraries x "
          f"{len({r['budget'] for r in rows})} budgets\n{'=' * 78}")
    hdr = f"  {'gate':<24s}{'acts on':>9s}{'dAUC vs BO':>12s}{'vs rand@cov':>13s}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for nm in dict.fromkeys(r["arm"] for r in rows):
        sel = [r for r in rows if r["arm"] == nm]
        c = float(np.mean([r["coverage"] for r in sel]))
        dd = float(np.mean([r["dauc"] for r in sel]))
        gaps = [r["vs_rand_at_coverage"] for r in sel
                if r.get("vs_rand_at_coverage") is not None]
        g = f"{np.mean(gaps):>+13.4f}" if gaps and not nm.startswith("maxmu_rand") else ""
        print(f"  {nm:<24s}{c:>8.0%}{dd:>+12.4f}{g:>13s}")
    print("\n  A gate is evidence only in the last column: the gap against a "
          "random gate\n  acting equally often. Anything positive in column "
          "three alone is the\n  selector being switched off, which is free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does free history seeding remove the regime an LLM chooser is paid for?

    python scripts/bad_start_gate.py --arms stl mt

Akke et al. (2025) compare LLM-based acquisition, agentic workflows and
classical BO on molecular and protein design. Their clearest molecular win for
the agentic workflow is not on a normal campaign -- it is on one *initialised in
a bad local minimum*, where the agent escapes and the statistical acquisition
functions do not. Their four campaigns are single-target, so the paper has no
related protein to borrow measurements from, and the question it leaves open is
the one this project is already set up to answer:

    the bad start is an initial design. If a *good* initial design can be bought
    for free from a relative's measured data, does the regime that pays the
    chooser still exist?

## The comparison

Two baselines, same rules, same everything else.

* `--start random` -- plain BO. Already measured: the raw rung of the quality
  ladder is exactly this run, so it is **read from disk, not recomputed**. If
  the stored artefact is missing the run happens here.
* `--start bad` -- the baseline designs are drawn as a low-scoring compound plus
  its nearest Tanimoto neighbours.

The free rules (`aux_top`, `aux_scaffold_nn`, `maxmin`) are identical in both.
They do not know where the baseline started, which is the point: a rule that
picks its own design is *immune* to the bad start by construction, and the
question is only how much that immunity is worth.

## What the numbers mean

`drop` is how much AUC the bad start costs plain BO, on the absolute scale.
`gain_bad` is what a rule recovers relative to it. The quantity to read is

    recovered = gain_bad / drop

at 1.0 the free rule has fully erased the bad start; above 1.0 it has done
better than never having had one. A rule can clear 1.0 because it is not
"repairing" the bad start at all -- it never takes it.

Note `drop` is bounded by what the *campaign* can undo: BO from a bad start
still has `budget - n_init` evaluations to recover, so a small drop is a
statement about the optimiser's resilience, not about the start being mild.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import init_headroom  # noqa: E402

RULES = ("aux_top", "aux_scaffold_nn", "maxmin")
BLOCKS = ("kiba_prkc", "kiba_cdk5", "kiba_lck")
#: Where the quality ladder stored its undegraded runs. Those *are* the
#: random-start condition at identical settings, so reusing them keeps this
#: script to the six runs it actually adds.
LADDER_DIR = Path("results/seed_quality")


def load_or_run(dest: Path, argv: list[str], *, force: bool) -> dict:
    if force or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if init_headroom.main(argv + ["--out", str(dest)]) != 0:
            raise SystemExit(f"run failed: {dest}")
    return json.loads(dest.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    p.add_argument("--arms", nargs="+", default=["stl"])
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--bad-quantile", type=float, default=0.25)
    p.add_argument("--out-dir", default="results/bad_start")
    p.add_argument("--out", default="results/bad_start_gate.json")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    t0 = time.time()
    common = ["--keep-censored", "--history-excludes-top",
              "--candidates", str(args.candidates),
              "--repeats", str(args.repeats),
              "--budget", str(args.budget),
              "--history", str(args.history)]

    rows = []
    for block in args.blocks:
        for arm in args.arms:
            base = ["--family", block, "--arm", arm] + common
            # The ladder's undegraded rung is the random-start condition at
            # these exact settings. Reuse it rather than paying for it twice --
            # but only if it carries the baseline's absolute level, which runs
            # stored before `baseline_auc_mean` existed do not. An older
            # artefact is left exactly as it is and the condition is re-run
            # here; nothing stored is rewritten.
            ladder = LADDER_DIR / f"{block}_raw_{arm}.json"
            reuse = (ladder.exists() and not args.force
                     and "baseline_auc_mean" in json.loads(
                         ladder.read_text(encoding="utf-8")))
            src = ladder if reuse else out_dir / f"{block}_random_{arm}.json"
            rnd = load_or_run(src, base, force=args.force)
            bad = load_or_run(out_dir / f"{block}_bad_{arm}.json",
                              base + ["--start", "bad",
                                      "--bad-quantile", str(args.bad_quantile)],
                              force=args.force)

            by_r = {r["label"]: r for r in rnd["rows"]}
            by_b = {r["label"]: r for r in bad["rows"]}
            drop = rnd["baseline_auc_mean"] - bad["baseline_auc_mean"]
            drop_sem = float(np.hypot(rnd["baseline_auc_sem"],
                                      bad["baseline_auc_sem"]))
            row = {
                "block": block, "arm": arm,
                "baseline_random": rnd["baseline_auc_mean"],
                "baseline_bad": bad["baseline_auc_mean"],
                "drop": drop, "drop_sem": drop_sem,
                "ceiling_random": by_r["hindsight ceiling"]["delta_mean"],
                "ceiling_bad": by_b["hindsight ceiling"]["delta_mean"],
                "gate_random_inert": (rnd.get("gate") or {}).get("inert"),
                "gate_bad_inert": (bad.get("gate") or {}).get("inert"),
                "audit_clean": bool(rnd["audit"]["clean"]
                                    and bad["audit"]["clean"]),
                "reused_ladder_run": bool(reuse),
                "artefacts": {"random": src.as_posix(),
                              "bad": (out_dir
                                      / f"{block}_bad_{arm}.json").as_posix()}}
            for r in RULES:
                if r in by_r:
                    row[f"{r}_random"] = by_r[r]["delta_mean"]
                    row[f"{r}_random_sem"] = by_r[r]["delta_sem"]
                if r in by_b:
                    row[f"{r}_bad"] = by_b[r]["delta_mean"]
                    row[f"{r}_bad_sem"] = by_b[r]["delta_sem"]
                    row[f"{r}_recovered"] = (by_b[r]["delta_mean"] / drop
                                             if drop > 0 else None)
            rows.append(row)
            print(f"  {block:<11s} {arm:<4s} drop={drop:+.3f}+-{drop_sem:.3f}  "
                  + "  ".join(f"{r}_bad={row.get(f'{r}_bad', float('nan')):+.3f}"
                              for r in ("aux_top", "aux_scaffold_nn"))
                  + f"   ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n    {'block':<11s}{'arm':>5s}{'drop':>9s}"
          f"{'aux_top|rnd':>13s}{'aux_top|bad':>13s}{'recovered':>11s}"
          f"{'inert|rnd':>11s}{'inert|bad':>11s}")
    print("    " + "-" * 84)
    for r in rows:
        rec = r.get("aux_top_recovered")
        print(f"    {r['block']:<11s}{r['arm']:>5s}{r['drop']:>+9.3f}"
              f"{r.get('aux_top_random', float('nan')):>+13.3f}"
              f"{r.get('aux_top_bad', float('nan')):>+13.3f}"
              f"{(f'{rec:.2f}x' if rec is not None else 'n/a'):>11s}"
              f"{str(r['gate_random_inert']):>11s}"
              f"{str(r['gate_bad_inert']):>11s}")

    # The headline, stated once: an LLM chooser is worth building only where the
    # ceiling still clears the best free rule. If the bad start is inert too,
    # the regime Akke et al. report a win in has been closed by a free rule.
    n = len(rows)
    inert_bad = sum(bool(r["gate_bad_inert"]) for r in rows)
    print(f"\n    gate says INERT on {inert_bad}/{n} cells from a bad start, "
          f"{sum(bool(r['gate_random_inert']) for r in rows)}/{n} from random")
    clean = all(r["audit_clean"] for r in rows)
    print(f"    audit clean on every cell: {clean}")

    report = {"blocks": args.blocks, "arms": args.arms,
              "candidates": args.candidates, "repeats": args.repeats,
              "budget": args.budget, "history": args.history,
              "bad_quantile": args.bad_quantile,
              "rows": rows, "audit_clean": bool(clean)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

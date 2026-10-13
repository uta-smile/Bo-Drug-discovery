"""Calibrate the avoidance-only arm to the pooled model's own avoidance, then
run it there instead of interpolating.

    python scripts/avoidance_calibration.py                 # report the plan
    python scripts/avoidance_calibration.py --run           # and run the cells
    python scripts/avoidance_calibration.py --run --iterate 1

``stl_avoid`` demotes historically measured compounds in the EI ranking by a
factor ``--avoid-weight W`` and reads no auxiliary values at all. Sweeping W
from 0 (near-total avoidance) to 1 (exactly ``stl``) traces out how much of the
pooled model's design-sensitivity avoidance alone can produce. The question
that sweep answers is not "can avoidance do this" but "can avoidance do this
*at the intensity the pooled model actually avoids at*", so W has to be set
from a measurement rather than chosen: the calibration target is ``pooled``'s
own measured ``history_selection_ratio`` in the same cell.

The sweep is on a six-point grid, and the calibrated W falls between grid
points. Reading the swing off the grid by linear interpolation gives an
estimate of a quantity that was never run -- defensible, but it is an estimate,
and the recovered fraction it produces depends on an interpolation rule that no
artefact records. This script does three things about that:

* states the rule, in one place: W is interpolated on the measured
  ratio-versus-W curve, averaged over the two history designs, since a single
  W has to serve both halves of the swing;
* writes down the interpolated prediction *before* running anything, so the
  direct runs can be read against it rather than replacing it silently;
* runs the calibrated W directly, and reports the achieved selection ratio
  next to the target it was calibrated to -- which is the only way to see that
  the interpolation landed where it claimed.

The swing is the quantity the mechanism claim rests on:

    swing(arm) = [recall(arm) - recall(stl)] withheld
               - [recall(arm) - recall(stl)] uniform

in **top-10 recall points**, the metric the C4 result and the rest of the
paper are stated in, and the headline number is
swing(stl_avoid at calibrated W) / swing(pooled). Both deltas are recomputed
here from the per-campaign `recall` vectors. A stored `delta_vs_stl` is the
*AUC* difference -- reading it as recall is how an earlier version of this
script reported pooled's 45%-coverage swing as 15.28 points where the C4
result is 29.07, from the very same runs. The AUC swing is carried in the
report beside the recall one, so the two can never be confused again, and the
configuration of the reference runs is printed with them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PY = sys.executable

GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DESIGNS = ("withheld", "uniform")
# The cell the avoidance experiments were run in: CYP, budget 111, 32 repeats
# per target. H=50 covers 45% of the library, H=100 covers 90%.
CELL = {"family": "cyp", "budget": 111, "batch": 10, "n_init": 10, "k": 10,
        "repeats": 32, "pile_frac": 0.02}
HISTORIES = (50, 100)
# Everything that has to match between two runs before their difference is a
# measurement rather than a comparison of two experiments. Printed with the
# swing, because a swing quoted without them is not checkable.
CONFIG_FIELDS = ("budget", "batch", "n_init", "k", "repeats", "history",
                 "history_excludes_top", "history_cluster", "acquisition",
                 "aux_quality_requested", "aux_quality_achieved", "xi", "beta",
                 "sd_scale", "pile_frac", "n_pool", "avoid_weight")


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arm_stats(art: dict, arm: str) -> dict:
    """Both metrics, recomputed from the per-campaign vectors.

    An artefact's stored ``delta_vs_stl`` is the **AUC** difference, not the
    recall difference -- `dense_block_campaign.py` builds it from
    ``acc[arm]["auc"]``. Reading that field as recall is how the first version
    of this script reported a pooled design swing of 15.28 points at 45%
    coverage where the C4 result is 29.07: the same runs, in different units.
    Nothing here reads it; the deltas below are formed from ``recall`` and
    ``auc`` by name, and ``check_delta_field`` asserts which is which.
    """
    a = art["arms"][arm]
    base = art["arms"]["stl"]
    rec = np.asarray(a["recall"], float) - np.asarray(base["recall"], float)
    auc = np.asarray(a["auc"], float) - np.asarray(base["auc"], float)
    return {"recall_mean": a["recall_mean"],
            "delta_recall": float(rec.mean() * 100.0),
            "delta_recall_sem": float(rec.std(ddof=1) / np.sqrt(len(rec)) * 100.0),
            "delta_auc": float(auc.mean()),
            "n_campaigns": int(len(rec)),
            "ratio": a.get("history_selection_ratio"),
            "ratio_sem": a.get("history_selection_ratio_sem"),
            "recalls": a.get("recall")}


def check_delta_field(art: dict, arm: str, stats: dict) -> str | None:
    """Pin down which metric the stored ``delta_vs_stl`` is, every time."""
    stored = art["arms"][arm].get("delta_vs_stl")
    if stored is None:
        return None
    if abs(stored - stats["delta_auc"]) < 1e-9:
        return "auc"
    if abs(stored - stats["delta_recall"] / 100.0) < 1e-9:
        return "recall"
    return "neither"


def swing(withheld: float, uniform: float) -> float:
    """What the history design is worth to an arm, in the units given.

    Both deltas must be in the same metric; the callers below pass recall
    points, and the AUC swing is carried alongside as a secondary column.
    """
    return withheld - uniform


def interpolate(xs, ys, target) -> float:
    """Where a monotone curve hits ``target``, by linear interpolation.

    ``np.interp`` needs an increasing x. The ratio rises with W in every cell
    measured here, but that is a property of the data, not a guarantee, so it
    is checked rather than assumed.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if not np.all(np.diff(xs) > 0):
        raise ValueError(f"ratio is not increasing in W: {list(xs)}")
    return float(np.interp(target, xs, ys))


def sweep_table(history: int) -> dict:
    """The measured grid for one coverage, plus the pooled targets."""
    rows = {}
    for design in DESIGNS:
        base = load(f"avoid_cyp_h{history}_{design}.json")
        if base is None:
            raise SystemExit(f"missing baseline avoid_cyp_h{history}_{design}.json")
        rows[design] = {
            "pooled": arm_stats(base, "pooled"),
            "stl": arm_stats(base, "stl"),
            "artefact": f"avoid_cyp_h{history}_{design}.json",
            "config": {k: base.get(k) for k in CONFIG_FIELDS},
            "grid": {},
        }
        rows[design]["delta_field_is"] = check_delta_field(
            base, "pooled", rows[design]["pooled"])
        for w in GRID:
            art = load(f"avoidw_cyp_h{history}_{design}_w{w:.2f}.json")
            if art is None:
                raise SystemExit(f"missing avoidw_cyp_h{history}_{design}_w{w:.2f}.json")
            rows[design]["grid"][w] = arm_stats(art, "stl_avoid")
    return rows


def calibrate(history: int, rows: dict) -> dict:
    """W such that the avoidance-only arm avoids as hard as pooled does."""
    target = float(np.mean([rows[d]["pooled"]["ratio"] for d in DESIGNS]))
    # One W must serve both designs, so the curve it is read off is the
    # average of the two. They differ by under 0.03 at every grid point here.
    ratios = [float(np.mean([rows[d]["grid"][w]["ratio"] for d in DESIGNS]))
              for w in GRID]
    w_star = interpolate(ratios, GRID, target)
    out = {
        "history": history,
        "coverage": 3 * history / rows["uniform"]["config"]["n_pool"],
        "pooled_ratio_withheld": rows["withheld"]["pooled"]["ratio"],
        "pooled_ratio_uniform": rows["uniform"]["pooled"]["ratio"],
        "target_ratio": target,
        "w_star": w_star,
        "grid_ratio": dict(zip(map(str, GRID), ratios)),
        "campaigns_per_cell": rows["uniform"]["pooled"]["n_campaigns"],
        "artefacts": {d: rows[d]["artefact"] for d in DESIGNS},
        "config": rows["uniform"]["config"],
        "stored_delta_field_is": rows["uniform"]["delta_field_is"],
    }
    # Both metrics, so that "the swing" can never again mean whichever field
    # was nearest to hand. Recall points are the headline, because that is the
    # metric the C4 result and the rest of the paper are stated in.
    for metric in ("recall", "auc"):
        key = f"delta_{metric}"
        swings = [swing(rows["withheld"]["grid"][w][key],
                        rows["uniform"]["grid"][w][key]) for w in GRID]
        # The swing curve is read at W*, which is what "interpolated estimate"
        # means. It is not monotone in general, so np.interp on W directly.
        predicted = float(np.interp(w_star, GRID, swings))
        pooled = swing(rows["withheld"]["pooled"][key],
                       rows["uniform"]["pooled"][key])
        out[metric] = {
            "grid_swing": dict(zip(map(str, GRID), swings)),
            "pooled_swing": pooled,
            "pooled_withheld": rows["withheld"]["pooled"][key],
            "pooled_uniform": rows["uniform"]["pooled"][key],
            "interpolated_swing": predicted,
            "interpolated_fraction": (predicted / pooled if pooled
                                      else float("nan")),
        }
    # Back-compatible aliases, in the headline metric.
    out["pooled_swing"] = out["recall"]["pooled_swing"]
    out["interpolated_swing"] = out["recall"]["interpolated_swing"]
    out["interpolated_fraction"] = out["recall"]["interpolated_fraction"]
    out["grid_swing"] = out["recall"]["grid_swing"]
    return out


def emit_macros(report: dict, tex_dir: Path) -> Path:
    """Render the measured calibration as LaTeX macros.

    Every macro name carries its metric. The first version of this analysis
    reported an AUC difference as recall points; a name like
    ``avoidRecallSwingHundred`` cannot be pasted into a recall sentence by
    mistake, and the AUC twin sits beside it so the two are visibly different
    quantities rather than one number of uncertain units.

    Only the *measured* direct runs are emitted. The interpolated estimates
    stay in the JSON: they were superseded the moment the runs existed, and a
    macro for them would invite quoting a prediction as a result.
    """
    names = {50: "Fifty", 100: "Hundred"}
    lines = ["% Generated by scripts/avoidance_calibration.py -- do not edit.",
             "% The auxiliary-value-free avoidance control (C4), measured at the",
             "% penalty W calibrated to the pooled model's own selection ratio.",
             ""]
    for history, suffix in names.items():
        entry = report["coverages"].get(str(history))
        if not entry or "direct" not in entry:
            continue
        cal, d = entry["calibration"], entry["direct"]
        for name, value in (
                ("avoidCoverage", f"${cal['coverage'] * 100:.0f}\\%$"),
                ("avoidHistory", f"{cal['history']}"),
                ("avoidW", f"${d['w']:.4f}$"),
                ("avoidTargetRatio", f"${cal['target_ratio']:.4f}$"),
                ("avoidAchievedRatio", f"${d['achieved_ratio']:.4f}$"),
                ("avoidCampaigns", f"{d['campaigns_per_cell']}"),
                ("avoidRecallWithheld", f"${d['recall']['delta_withheld']:+.2f}$"),
                ("avoidRecallUniform", f"${d['recall']['delta_uniform']:+.2f}$"),
                ("avoidRecallSwing", f"${d['recall']['measured_swing']:+.2f}$"),
                ("avoidRecallFrac", f"${d['recall']['measured_fraction']:.0%}$"
                                    .replace("%", "\\%")),
                ("avoidAucSwing", f"${d['auc']['measured_swing']:+.4f}$"),
                ("avoidAucFrac", f"${d['auc']['measured_fraction']:.0%}$"
                                 .replace("%", "\\%")),
                ("pooledRecallSwing", f"${cal['recall']['pooled_swing']:+.2f}$"),
                ("pooledRecallWithheld",
                 f"${cal['recall']['pooled_withheld']:+.2f}$"),
                ("pooledRecallUniform",
                 f"${cal['recall']['pooled_uniform']:+.2f}$"),
        ):
            lines.append(f"\\newcommand{{\\{name}{suffix}}}{{{value}}}")
        lines.append("")
    path = tex_dir / "macros_avoidance.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_cell(history: int, design: str, w: float, work: Path, *, force=False):
    """One direct run at the calibrated W, with the sweep's own flags."""
    out = work / f"avoidcal_cyp_h{history}_{design}_w{w:.4f}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8")), out
    argv = [PY, str(ROOT / "scripts" / "dense_block_campaign.py"),
            "--family", CELL["family"], "--history", str(history),
            "--budget", str(CELL["budget"]), "--batch", str(CELL["batch"]),
            "--n-init", str(CELL["n_init"]), "--k", str(CELL["k"]),
            "--repeats", str(CELL["repeats"]),
            "--pile-frac", str(CELL["pile_frac"]),
            # The sweep ran under the destroyed-value null, so the avoidance
            # arm's picks cannot be confused with transfer. `stl_avoid` reads
            # no auxiliary values either way; this keeps the two comparable.
            "--aux-quality", "0",
            *(["--history-excludes-top"] if design == "withheld" else []),
            "--avoid-weight", f"{w:.4f}",
            "--audit", "--arms", "stl", "stl_avoid",
            "--out", str(out)]
    log = out.with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc.returncode or not out.exists():
        raise SystemExit(f"run failed: {' '.join(argv)}\n    see {log}")
    return json.loads(out.read_text(encoding="utf-8")), out


def direct(cal: dict, work: Path, *, force=False) -> dict:
    """Run the calibrated W and measure what it actually did."""
    w = cal["w_star"]
    ran = {d: run_cell(cal["history"], d, w, work, force=force) for d in DESIGNS}
    runs = {d: art for d, (art, _) in ran.items()}
    stats = {d: arm_stats(runs[d], "stl_avoid") for d in DESIGNS}
    out = {
        "w": w,
        "achieved_ratio_withheld": stats["withheld"]["ratio"],
        "achieved_ratio_uniform": stats["uniform"]["ratio"],
        "achieved_ratio": float(np.mean([stats[d]["ratio"] for d in DESIGNS])),
        "campaigns_per_cell": stats["uniform"]["n_campaigns"],
        "audit_clean": all(runs[d]["audit"]["clean"] for d in DESIGNS),
        "artefacts": {d: path.relative_to(ROOT).as_posix()
                      for d, (_, path) in ran.items()},
        # The history draw is a function of the seed, the held-out target and
        # the library, so the calibrated run and the run it is compared with
        # should be on the same historical compounds. Recorded, not assumed.
        "history_digest": {d: (runs[d].get("digest") or {}).get("history")
                           for d in DESIGNS},
    }
    # The campaigns are paired across designs only through the arm; the
    # per-campaign vectors stay in the artefacts so a reader can test the
    # swing rather than take the difference of two means on faith.
    for metric in ("recall", "auc"):
        key = f"delta_{metric}"
        measured = swing(stats["withheld"][key], stats["uniform"][key])
        pooled = cal[metric]["pooled_swing"]
        out[metric] = {
            "delta_withheld": stats["withheld"][key],
            "delta_uniform": stats["uniform"][key],
            "measured_swing": measured,
            "measured_fraction": measured / pooled if pooled else float("nan"),
            "interpolation_error": measured - cal[metric]["interpolated_swing"],
        }
    out["delta_withheld"] = out["recall"]["delta_withheld"]
    out["delta_uniform"] = out["recall"]["delta_uniform"]
    out["measured_swing"] = out["recall"]["measured_swing"]
    out["measured_fraction"] = out["recall"]["measured_fraction"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true",
                   help="run the calibrated W directly (2 cells per coverage, "
                        "about 2 minutes each)")
    p.add_argument("--iterate", type=int, default=0,
                   help="secant steps to take if the achieved ratio misses "
                        "the target by more than --tol")
    p.add_argument("--tol", type=float, default=0.01,
                   help="how close the achieved selection ratio has to land")
    p.add_argument("--work", type=Path, default=RESULTS,
                   help="where the direct runs are written")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path,
                   default=RESULTS / "avoidance_calibration.json")
    p.add_argument("--tex-dir", type=Path, default=None,
                   help="also render the measured result as LaTeX macros "
                        "(report/generated)")
    args = p.parse_args(argv)

    t0 = time.time()
    report = {"cell": CELL, "grid": list(GRID), "coverages": {}}
    for history in HISTORIES:
        rows = sweep_table(history)
        cal = calibrate(history, rows)
        print(f"\n=== history {history} per auxiliary "
              f"({cal['coverage']:.0%} library coverage) ===")
        cfg = cal["config"]
        print(f"  reference runs: {cal['artefacts']['withheld']} / "
              f"{cal['artefacts']['uniform']}")
        print(f"    budget {cfg['budget']}, batch {cfg['batch']}, "
              f"n_init {cfg['n_init']}, top-{cfg['k']}, "
              f"{cfg['repeats']} repeats x {cal['campaigns_per_cell'] // cfg['repeats']} "
              f"held-out targets = {cal['campaigns_per_cell']} campaigns, "
              f"pool {cfg['n_pool']}")
        print(f"    history {cfg['history']}x3 drawn "
              f"{'clear of the top-k' if cfg['history_excludes_top'] else 'uniformly'}"
              f" / uniformly, auxiliary q={cfg['aux_quality_requested']} "
              f"(achieved Spearman {cfg['aux_quality_achieved']:+.4f}), "
              f"xi={cfg['xi']}, beta={cfg['beta']}, sd_scale={cfg['sd_scale']}")
        print(f"    the artefact's stored delta_vs_stl is the "
              f"{cal['stored_delta_field_is'].upper()} difference; the swings "
              "below are recomputed from the recall vectors")
        print(f"  pooled avoids at ratio {cal['target_ratio']:.4f} "
              f"(withheld {cal['pooled_ratio_withheld']:.4f}, "
              f"uniform {cal['pooled_ratio_uniform']:.4f}); its swing is "
              f"{cal['recall']['pooled_withheld']:+.2f} - "
              f"({cal['recall']['pooled_uniform']:+.2f}) = "
              f"{cal['pooled_swing']:+.2f} recall points "
              f"({cal['auc']['pooled_swing']:+.4f} AUC)")
        print(f"  {'W':>6} {'ratio':>8} {'swing(recall)':>14} {'swing(AUC)':>12}")
        for w in GRID:
            print(f"  {w:6.2f} {cal['grid_ratio'][str(w)]:8.4f} "
                  f"{cal['recall']['grid_swing'][str(w)]:+14.2f} "
                  f"{cal['auc']['grid_swing'][str(w)]:+12.4f}")
        print(f"  calibrated W = {cal['w_star']:.4f} -> interpolated swing "
              f"{cal['interpolated_swing']:+.2f} recall points "
              f"({cal['interpolated_fraction']:.0%} of pooled)")
        entry = {"calibration": cal}

        if args.run:
            d = direct(cal, args.work, force=args.force)
            steps = [d]
            for _ in range(args.iterate):
                miss = d["achieved_ratio"] - cal["target_ratio"]
                if abs(miss) <= args.tol:
                    break
                # Secant on the measured (W, ratio) pairs, falling back to the
                # grid neighbour when only one direct point exists.
                xs = [s["achieved_ratio"] for s in steps]
                ws = [s["w"] for s in steps]
                if len(steps) == 1:
                    j = int(np.argmin([abs(cal["grid_ratio"][str(w)]
                                           - cal["target_ratio"]) for w in GRID]))
                    xs = xs + [cal["grid_ratio"][str(GRID[j])]]
                    ws = ws + [GRID[j]]
                order = np.argsort(xs)
                w_next = float(np.interp(cal["target_ratio"],
                                         np.asarray(xs)[order],
                                         np.asarray(ws)[order]))
                if abs(w_next - d["w"]) < 1e-4:
                    break
                cal_next = dict(cal, w_star=w_next)
                d = direct(cal_next, args.work, force=args.force)
                steps.append(d)
            entry["direct"] = d
            entry["direct_steps"] = steps
            print(f"  direct run at W={d['w']:.4f}: achieved ratio "
                  f"{d['achieved_ratio']:.4f} "
                  f"(withheld {d['achieved_ratio_withheld']:.4f}, "
                  f"uniform {d['achieved_ratio_uniform']:.4f}), "
                  f"swing {d['recall']['delta_withheld']:+.2f} - "
                  f"({d['recall']['delta_uniform']:+.2f}) = "
                  f"{d['measured_swing']:+.2f} recall points "
                  f"({d['measured_fraction']:.0%} of pooled; "
                  f"{d['auc']['measured_fraction']:.0%} by AUC)")
            print(f"  interpolation error: swing "
                  f"{d['recall']['interpolation_error']:+.2f} "
                  f"recall points, ratio "
                  f"{d['achieved_ratio'] - cal['target_ratio']:+.4f}; "
                  f"audit clean: {d['audit_clean']}")
        report["coverages"][str(history)] = entry

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({time.time() - t0:.0f}s)")
    if args.tex_dir:
        print(f"wrote {emit_macros(report, args.tex_dir)}")
    if not args.run:
        print("nothing was run; pass --run to measure the calibrated W "
              "instead of interpolating it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

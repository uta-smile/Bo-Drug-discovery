"""Are the uniform-history cells and their two nulls really a matched pair?

    python scripts/history_pairing_audit.py --dry-run
    python scripts/history_pairing_audit.py --jobs 6
    python scripts/history_pairing_audit.py --jobs 6 --only cyp,jak

The six-family design comparison and the permutation control are *paired*
statistics: every cell is read as a difference between a run with measured
auxiliary activities (``winleak_*``) and the same run with those activities
destroyed (``sdrefu_*_q0``) or permuted (``sdrefu_*_shuf``). The pairing is
only meaningful if the three runs assayed the *same* historical compounds from
the same starting designs, and differ in nothing but the auxiliary column.

What the stored artefacts could show until now was that the three runs agreed
on summary statistics of the history -- its size, and ``reach``, the mean
nearest-neighbour similarity it induces. Those are much weaker than equality:
two entirely different draws of 150 compounds out of 335 land at the same reach
routinely. Nothing compared the drawn rows themselves, because nothing stored
them.

So this re-runs all twelve cells under all three auxiliary conditions, with
`RunDigest` (`dense_block_campaign.py`) recording a content hash of the drawn
history indices, the values handed to the arms at those indices, and the
initial designs. Three things then have to hold, and each is a different way
for the pairing to be broken:

1. every re-run reproduces its committed artefact's per-campaign recall and
   AUC vectors exactly -- otherwise the digests describe some *other* run than
   the one the paper quotes;
2. within a cell, the three conditions agree bit for bit on the history digest
   and the initial-design digest -- the masks are identical, not merely
   similar;
3. within a cell, the three auxiliary-value digests are pairwise distinct --
   the nulls actually took effect, which is the failure mode that equality
   checks alone would silently pass.

Cells come from ``run_window_experiments.plan("leak")``, so the budgets are
re-derived from the data here exactly as they were when the artefacts were
written, rather than copied into a second list that can drift.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

# The auxiliary condition is the *only* thing that may differ between the three
# runs of a cell. `real` carries no flag at all, which is how `winleak_*` was
# written -- passing `--aux-quality 1.0` reproduces it bit for bit, but the
# audit is about the stored artefacts, so it runs them the way they were run.
CONDITIONS = {
    "real": {"flags": [], "artefact": "winleak_{family}_b{budget}.json",
             "doc": "measured auxiliary activities"},
    "q0": {"flags": ["--aux-quality", "0"],
           "artefact": "sdrefu_{family}_b{budget}_q0.json",
           "doc": "Gaussian null: information and shape destroyed"},
    "shuf": {"flags": ["--aux-shuffle"],
             "artefact": "sdrefu_{family}_b{budget}_shuf.json",
             "doc": "permutation null: information destroyed, shape kept"},
}


def cells(pile_frac: float = 0.02):
    """The twelve uniform-history cells, re-derived rather than hardcoded."""
    from run_window_experiments import EXPERIMENTS, plan

    out = []
    for argv, path, label in plan("leak", EXPERIMENTS["leak"], pile_frac):
        family = argv[argv.index("--family") + 1]
        budget = int(argv[argv.index("--budget") + 1])
        out.append({"family": family, "budget": budget, "label": label,
                    "argv": argv, "winleak": path})
    return out


def run_one(cell, condition, work: Path, *, force=False):
    """One re-run. Returns the parsed artefact, or None if it failed."""
    out = work / f"{condition}_{cell['family']}_b{cell['budget']}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    argv = list(cell["argv"])
    argv[argv.index("--out") + 1] = str(out)
    argv += CONDITIONS[condition]["flags"]
    log = out.with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if rc.returncode or not out.exists():
        print(f"    FAILED {condition} {cell['label']} -- see {log}")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def vectors_match(fresh: dict, stored: dict) -> tuple[bool, list[str]]:
    """Per-campaign recall and AUC, arm by arm, exactly."""
    problems = []
    if set(fresh["arms"]) != set(stored["arms"]):
        problems.append(f"arms differ: {sorted(fresh['arms'])} vs "
                        f"{sorted(stored['arms'])}")
    for arm in sorted(set(fresh["arms"]) & set(stored["arms"])):
        for key in ("recall", "auc"):
            a, b = fresh["arms"][arm].get(key), stored["arms"][arm].get(key)
            if a != b:
                n = sum(1 for x, y in zip(a or [], b or []) if x != y)
                problems.append(f"{arm}.{key}: {n} of {len(b or [])} campaigns "
                                "differ")
    if fresh["reach"] != stored["reach"]:
        problems.append(f"reach {fresh['reach']} != {stored['reach']}")
    return not problems, problems


def audit_cell(cell, fresh: dict[str, dict]) -> dict:
    """The three checks, for one cell. ``fresh`` maps condition -> artefact."""
    rec = {"family": cell["family"], "budget": cell["budget"],
           "label": cell["label"], "problems": [], "digests": {}}

    # 1. the re-runs are the stored runs
    for cond, art in fresh.items():
        stored_path = RESULTS / CONDITIONS[cond]["artefact"].format(**cell)
        if not stored_path.exists():
            rec["problems"].append(f"{cond}: no stored artefact "
                                   f"{stored_path.name}")
            continue
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        ok, problems = vectors_match(art, stored)
        rec[f"reproduces_{cond}"] = ok
        rec["problems"] += [f"{cond} vs {stored_path.name}: {p}"
                            for p in problems]
        if not art.get("audit", {}).get("clean", False):
            rec["problems"].append(f"{cond}: audit not clean")

    for cond, art in fresh.items():
        rec["digests"][cond] = {k: art["digest"][k]
                                for k in ("history", "aux_value",
                                          "init_design")}
        rec["digests"][cond]["cells"] = art["digest"]["cells"]

    conds = sorted(fresh)
    # 2. identical masks and starting points
    for stream in ("history", "init_design"):
        vals = {rec["digests"][c][stream] for c in conds}
        rec[f"{stream}_identical"] = len(vals) == 1
        if len(vals) != 1:
            rec["problems"].append(
                f"{stream} digests differ across conditions: "
                + ", ".join(f"{c}={rec['digests'][c][stream][:12]}"
                            for c in conds))
    # 3. the nulls actually changed the values
    vals = [rec["digests"][c]["aux_value"] for c in conds]
    rec["aux_values_distinct"] = len(set(vals)) == len(vals)
    if not rec["aux_values_distinct"]:
        same = [f"{a}={b[:12]}" for a, b in zip(conds, vals)]
        rec["problems"].append("auxiliary-value digests not pairwise "
                               "distinct: " + ", ".join(same))
    # Equal cell counts is implied by equal history digests (the cell key is
    # hashed too), but reported because a reader checks it first.
    counts = {rec["digests"][c]["cells"] for c in conds}
    rec["campaign_cells"] = counts.pop() if len(counts) == 1 else sorted(counts)
    rec["clean"] = not rec["problems"]
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=ROOT / "build" / "pairing_audit",
                   help="where the re-runs are written. Not results/: these "
                        "are verification copies of artefacts already there, "
                        "and build/ is already ignored by git")
    p.add_argument("--jobs", type=int, default=4,
                   help="cells to run at once; each is single-threaded enough "
                        "that 4-6 is a real speedup")
    p.add_argument("--only", default="",
                   help="comma-separated families, for a partial check")
    p.add_argument("--force", action="store_true",
                   help="re-run even where a re-run is already present")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", type=Path,
                   default=RESULTS / "history_pairing_audit.json")
    args = p.parse_args(argv)

    plan_cells = cells()
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        plan_cells = [c for c in plan_cells if c["family"] in want]
    jobs = [(c, cond) for c in plan_cells for cond in CONDITIONS]
    print(f"{len(plan_cells)} cells x {len(CONDITIONS)} auxiliary conditions "
          f"= {len(jobs)} runs")
    for cond, spec in CONDITIONS.items():
        print(f"    {cond:<6s} {spec['doc']:<48s} -> {spec['artefact']}")
    if args.dry_run:
        for c in plan_cells:
            print(f"    would run {c['label']}")
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = {}

    def task(job):
        c, cond = job
        art = run_one(c, cond, args.work, force=args.force)
        print(f"    [{len(done) + 1:>2}/{len(jobs)}] {cond:<5s} {c['label']}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        done[(c["family"], c["budget"], cond)] = art
        return art

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(task, jobs))

    records, failed = [], []
    for c in plan_cells:
        fresh = {cond: done[(c["family"], c["budget"], cond)]
                 for cond in CONDITIONS
                 if done.get((c["family"], c["budget"], cond)) is not None}
        if len(fresh) != len(CONDITIONS):
            failed.append(c["label"])
            continue
        records.append(audit_cell(c, fresh))

    width = max((len(r["label"]) for r in records), default=10)
    print(f"\n{'cell':<{width}} {'reruns=stored':>13} {'same history':>13} "
          f"{'same init':>10} {'nulls differ':>13}")
    for r in records:
        repro = all(r.get(f"reproduces_{c}", False) for c in CONDITIONS)
        print(f"{r['label']:<{width}} {str(repro):>13} "
              f"{str(r['history_identical']):>13} "
              f"{str(r['init_design_identical']):>10} "
              f"{str(r['aux_values_distinct']):>13}")
        for problem in r["problems"]:
            print(f"    ! {problem}")

    # Which cells share a draw. The history is a function of the seed, the
    # held-out target, the repeat and the library -- never of the budget -- so
    # the two budget cells of a family are expected to come out on the same
    # historical compounds from the same initial designs. That is the concrete
    # form of "two budgets on one family are not independent", and it is why
    # the cross-family statistics average the two cells into one family value
    # instead of counting twelve. Printed rather than asserted: if a future
    # sweep varies the history with the budget, the grouping says so.
    shared: dict[str, list[str]] = {}
    for r in records:
        key = r["digests"]["real"]["history"]
        shared.setdefault(key, []).append(f"{r['family']} b{r['budget']}")
    print("\ncells sharing one history draw:")
    for key, cells_ in sorted(shared.items(), key=lambda kv: kv[1]):
        print(f"    {key[:12]}  {', '.join(cells_)}")

    clean = [r for r in records if r["clean"]]
    verdict = {
        "shared_history_groups": {k[:16]: v for k, v in shared.items()},
        "cells": len(records), "clean": len(clean),
        "failed_runs": failed,
        "all_clean": len(clean) == len(records) and not failed,
        "conditions": {k: v["doc"] for k, v in CONDITIONS.items()},
        "campaigns_per_cell": sorted({r["campaign_cells"] for r in records
                                      if isinstance(r["campaign_cells"], int)}),
        "records": records,
    }
    args.out.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\n{len(clean)}/{len(records)} cells clean"
          + (f"; {len(failed)} runs failed: {', '.join(failed)}" if failed
             else "")
          + f"  ({time.time() - t0:.0f}s)")
    print(f"wrote {args.out}")
    return 0 if verdict["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

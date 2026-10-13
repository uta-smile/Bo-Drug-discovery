"""Re-run stored campaign cells with the runtime audit on, and check they
reproduce.

    python scripts/rerun_audited.py "results/sdscale_*.json" --out-dir results/audited
    python scripts/rerun_audited.py "results/sdscale_*.json" --dry-run

``scripts/experiment_inventory.py`` found that the matched-uncertainty sweep --
the evidence for the paper's claim that a single-task model at a reduced
uncertainty scale gains as much as a randomized auxiliary does -- was run
without ``--audit``. Its artefacts carry ``enabled: false``, which prints as
``clean: false`` and is indistinguishable at a glance from an audit that ran and
failed. The same mistake was caught once before, on the cross-family cells.

A single-arm run is not exempt. The audit's per-cell checks -- that the training
rows are exactly the budget, that the evaluation rows were never trained on,
that the initial design is the one that was handed in -- apply to one arm just
as much as to three.

This rebuilds each command line **from the artefact's own recorded parameters**
rather than from a launch script or from memory, re-runs it with ``--audit``
into a separate directory, and then compares the new numbers against the stored
ones. Both halves matter: the audit says the run was valid, and the comparison
says it is the same run. Stored artefacts are never overwritten.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Fields that map straight onto a flag of the same name.
SCALAR = (("budget", "--budget"), ("batch", "--batch"), ("k", "--k"),
          ("history", "--history"), ("repeats", "--repeats"),
          ("acquisition", "--acquisition"), ("pile_frac", "--pile-frac"),
          ("sd_scale", "--sd-scale"), ("history_cluster", "--history-cluster"))

# Every key an artefact can carry that describes how the run was configured,
# and which this rebuilder knows how to turn back into a flag. The first
# version of this script omitted `aux_quality_requested`, so the two q = 0
# cells were silently re-run with *real* auxiliary data; they failed the
# comparison, which is how it was caught, but a silent omission should not
# depend on a comparison to be noticed. Anything configuring the run that is
# not listed here makes the rebuild refuse.
HANDLED = {
    "family", "budget", "batch", "k", "history", "repeats", "acquisition",
    "pile_frac", "sd_scale", "history_cluster", "history_excludes_top",
    "aux_quality_requested", "scaffold_split", "seeding", "arms",
}

# Outputs and diagnostics. These describe what happened, not what was asked
# for, so they are not part of a command line.
DERIVED = {
    "audit", "n_pool", "per_target", "provenance", "reach",
    "targets_improving", "aux_quality_achieved", "rho", "rho_eff",
    "block_provenance", "coupling", "notes",
}


def argv_for(d: dict, out: Path) -> list[str]:
    unknown = sorted(set(d) - HANDLED - DERIVED)
    if unknown:
        raise SystemExit(
            f"this artefact records settings the rebuilder does not know how "
            f"to turn back into flags: {', '.join(unknown)}. Add them to "
            f"SCALAR/HANDLED (or to DERIVED if they are outputs) rather than "
            f"letting the re-run quietly use a default for them.")
    prov = d.get("provenance") or {}
    src = prov.get("source")
    if not src:
        raise SystemExit("artefact carries no provenance.source; its command "
                         "line cannot be rebuilt honestly.")
    argv = [PY, str(ROOT / "scripts" / "dense_block_campaign.py"),
            "--source", src, "--family", str(d["family"])]
    for key, flag in SCALAR:
        v = d.get(key)
        if v is not None:
            argv += [flag, str(v)]
    if prov.get("censoring") in (None, "none", "disabled"):
        argv.append("--keep-censored")
    if d.get("history_excludes_top"):
        argv.append("--history-excludes-top")
    if prov.get("blocks_from"):
        argv += ["--blocks-from", str(prov["blocks_from"])]
    if prov.get("seed_pool") and prov["seed_pool"] != "random":
        argv += ["--seed-pool", str(prov["seed_pool"])]
    if d.get("scaffold_split"):
        argv.append("--scaffold-split")
    # The one that was missing. `q = 1` happens to be equivalent to omitting
    # the flag, so leaving it out reproduced the measured-panel cells exactly
    # and only the randomized ones differed -- a failure mode that looks like
    # a data problem and is a rebuild problem.
    q = d.get("aux_quality_requested")
    if q is not None:
        if q == "shuffled":
            argv.append("--aux-shuffle")
        else:
            argv += ["--aux-quality", str(q)]
    arms = sorted(d.get("arms") or {})
    if arms:
        argv += ["--arms", *arms]
    argv += ["--audit", "--out", str(out)]
    return argv


def compare(old: dict, new: dict) -> list[str]:
    """Every per-arm summary number, to the stored precision."""
    bad = []
    for arm, a in (old.get("arms") or {}).items():
        b = (new.get("arms") or {}).get(arm)
        if b is None:
            bad.append(f"arm {arm} missing from the re-run")
            continue
        for field in ("auc_mean", "recall_mean", "auc_sem"):
            x, y = a.get(field), b.get(field)
            if x is None or y is None:
                continue
            if abs(float(x) - float(y)) > 1e-9:
                bad.append(f"{arm}.{field}: stored {x!r}, re-run {y!r}")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", help="glob over stored artefacts, quoted")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "results" / "audited")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    files = sorted(Path(f) for f in globmod.glob(args.pattern))
    if not files:
        raise SystemExit(f"nothing matches {args.pattern!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0, failures, reproduced = time.time(), [], 0
    for n, f in enumerate(files, 1):
        old = json.loads(f.read_text(encoding="utf-8"))
        out = args.out_dir / f.name
        cmd = argv_for(old, out)
        if args.dry_run:
            print("  " + " ".join(cmd[1:]))
            continue
        log = out.with_suffix(".log")
        with log.open("w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not out.exists():
            failures.append(f"{f.name}: run failed, see {log}")
            print(f"  [{n}/{len(files)}] {f.name}  FAILED")
            continue
        new = json.loads(out.read_text(encoding="utf-8"))
        audit = new.get("audit") or {}
        diffs = compare(old, new)
        ok = audit.get("enabled") and audit.get("clean") and not diffs
        reproduced += bool(ok)
        if not ok:
            failures.append(f"{f.name}: audit={audit.get('enabled')}/"
                            f"{audit.get('clean')} " + "; ".join(diffs))
        print(f"  [{n}/{len(files)}] {f.name}  "
              f"audit {'clean' if audit.get('clean') else 'NOT CLEAN'}, "
              f"{'reproduces' if not diffs else 'DIFFERS'}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    if args.dry_run:
        return 0
    print(f"\n  {reproduced}/{len(files)} re-ran clean and reproduced the "
          f"stored numbers exactly, in {time.time() - t0:.0f}s.")
    for msg in failures:
        print(f"  PROBLEM {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

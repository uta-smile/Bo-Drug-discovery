"""Assemble the evidence package that accompanies the manuscript.

    python scripts/build_evidence.py              # build into evidence/
    python scripts/build_evidence.py --out /tmp/e --zip

A reviewer cannot confirm a check from a description of it. This collects the
things needed to re-run every check in the reproducibility appendix and to
disagree with its conclusions:

* the code -- ``src/molbo`` and ``scripts/``, so the audit scripts and the
  campaign driver they rebuild command lines for are both present;
* the inputs -- every campaign artefact the manuscript draws a number from,
  and the block CSVs the preprocessing claims are computed on;
* the outputs -- the captured stdout and exit status of each check, produced
  by this script at build time rather than transcribed;
* ``MANIFEST.sha256``, so the files a reviewer runs can be shown to be the
  files that produced the reported output.

The exit status of each check is recorded and the build fails if any check
fails, so a package cannot be shipped asserting that checks pass when they do
not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (name, argv). Each is run from the package root so that the paths it prints
# are the paths a reviewer will have.
CHECKS = (
    ("preprocessing", ["scripts/preprocessing_audit.py",
                       "--json", "results/preprocessing_audit.json"]),
    ("inventory", ["scripts/experiment_inventory.py", "--cited-only",
                   "--json", "results/experiment_inventory.json"]),
    ("degradation", ["scripts/degradation_audit.py", "--n", "335", "433",
                     "--json", "results/degradation_audit.json"]),
    ("tables", ["scripts/regenerate_tables.py"]),
    # The summaries are the pipeline the tables and the abstract both render
    # from, so they are checked in the package rather than described in it.
    ("summaries", ["scripts/summaries.py", "--json", "results/summaries.json",
                   "--tex-dir", "report/generated"]),
    ("block_level_mt", ["scripts/block_level.py", "results/winleak_*.json",
                        "--arm", "mt", "--against", "results/win32_*.json"]),
    ("block_level_pooled", ["scripts/block_level.py", "results/winleak_*.json",
                            "--arm", "pooled",
                            "--against", "results/win32_*.json"]),
)

# Artefact groups the manuscript draws numbers from, plus the derived files.
ARTEFACT_GLOBS = (
    "results/auxq_*.json", "results/sdref_*.json", "results/sdscale_*.json",
    "results/shuf_*.json", "results/win32_*.json", "results/winleak_*.json",
    "results/xfam_seed/*.json", "results/xfam_seed_excl/*.json",
    "results/xfam_transfer/*.json", "results/xfam_transfer_excl/*.json",
    "results/audited/*.json",
    "results/transfer_rule_*.json", "results/kiba_block_rho.json",
    "results/cross_family_scan.json", "results/cross_family_blocks.json",
    "results/dockstring_audit.json",
    "results/selection_diagnosis_*.json", "results/scale_protocol_*.json",
    "results/null_mechanism_*.json", "results/mech_*.json",
    "results/summaries.json", "results/experiment_inventory.json",
    "results/preprocessing_audit.json", "results/degradation_audit.json",
)

README = """# Evidence package

Everything needed to re-run the checks in the reproducibility appendix of
*When Auxiliary Data Help Molecular Bayesian Optimization*, and to disagree
with them.

Nothing here is transcribed. `outputs/` was produced by running the commands
below at build time; `MANIFEST.sha256` covers every file in this package.

## Layout

    scripts/            the audit scripts and the campaign driver
    src/molbo/          the library they import (surrogate, metrics, audit)
    results/            every campaign artefact the manuscript cites
    results/audited/    re-runs of the artefacts that were executed without
                        the runtime audit; same parameters, audit enabled
    data/bindingdb/     the block CSVs the preprocessing claims are computed on
    outputs/            captured stdout and exit status of each check
    MANIFEST.sha256     sha256 of every file above

## Re-running

Requires Python with numpy, scipy, pandas and rdkit. From this directory:

{commands}

Each exits nonzero on a mismatch.

## What each check establishes

| check | input | what it verifies |
|---|---|---|
| `preprocessing` | the block CSVs | the twenty quantitative statements in the Data section: potency transform, censoring spikes and the doses they correspond to, modal share per column, compounds identical on all four targets, survival after the filter, task correlation before and after |
| `inventory` | every campaign artefact | seeds per target x held-out targets equals the stored per-campaign array length; artefacts within a group agree on protocol; groups recorded under both history designs are split |
| `degradation` | the degradation operator | the two closed forms for the variance ratio and the achieved correlation describe the implementation; the achieved values are reported as mean and as per-draw spread |
| `tables` | the stored artefacts | the null-control, history-design and decision-rule tables, with zero counts, tie counts and exactness for every signed-rank test |
| `block_level_*` | the window grid | the between-design contrast at campaign, configuration and block level, with the signed-rank floor |

## Known limitations of this package

* The campaign artefacts are included, not the raw TDC BindingDB dump
  (839 MB), which is a public download. The block CSVs cut from it are here.
* `rerun_audited.py` is included but re-running it needs that dump; its
  output is in `outputs/` and its products in `results/audited/`.
* The DOCKSTRING and KIBA analyses in the appendix draw on artefacts that are
  included, but their upstream libraries are not.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path, pattern="*") -> int:
    n = 0
    for f in sorted(src.rglob(pattern)):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "evidence")
    p.add_argument("--zip", action="store_true",
                   help="also write <out>.zip")
    args = p.parse_args(argv)

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"building into {out}\n")

    n = copy_tree(ROOT / "scripts", out / "scripts", "*.py")
    print(f"  scripts/           {n:>4} files")
    n = copy_tree(ROOT / "src" / "molbo", out / "src" / "molbo", "*.py")
    print(f"  src/molbo/         {n:>4} files")

    n = 0
    for pattern in ARTEFACT_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            target = out / f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    print(f"  results/           {n:>4} artefacts")

    n = copy_tree(ROOT / "data" / "bindingdb", out / "data" / "bindingdb",
                  "*.csv")
    print(f"  data/bindingdb/    {n:>4} block CSVs")

    # -- run each check inside the package ---------------------------------
    (out / "outputs").mkdir()
    statuses, failed = {}, []
    print()
    for name, cmd in CHECKS:
        t0 = time.time()
        r = subprocess.run([PY, "-I", *cmd], cwd=out, capture_output=True,
                           text=True)
        body = r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        (out / "outputs" / f"{name}.txt").write_text(
            f"$ python {' '.join(cmd)}\n"
            f"# exit status: {r.returncode}\n\n{body}", encoding="utf-8")
        statuses[name] = r.returncode
        if r.returncode != 0:
            failed.append(name)
        print(f"  {name:<20}exit {r.returncode}   "
              f"({time.time() - t0:.1f}s)")

    # The audited re-run is reported from the main tree; it needs the raw dump.
    src_log = ROOT / "results" / "audited"
    if src_log.exists():
        logs = sorted(src_log.glob("*.log"))
        if logs:
            (out / "outputs" / "rerun_audited.txt").write_text(
                "Re-runs of the artefacts originally executed without the "
                "runtime audit.\nEach command line was rebuilt from the "
                "artefact's own recorded parameters.\nProducts are in "
                "results/audited/; per-run logs follow.\n\n"
                + "\n\n".join(f"=== {f.name} ===\n"
                              + f.read_text(encoding="utf-8", errors="replace")
                              for f in logs),
                encoding="utf-8")
            print(f"  {'rerun_audited':<20}{len(logs)} logs copied")

    cmds = "\n".join(f"    python {' '.join(c)}" for _, c in CHECKS)
    (out / "README.md").write_text(README.format(commands=cmds),
                                   encoding="utf-8")

    # -- manifest last, so it covers the outputs too -----------------------
    lines, total = [], 0
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.sha256":
            lines.append(f"{sha256(f)}  {f.relative_to(out).as_posix()}")
            total += f.stat().st_size
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    print(f"\n  MANIFEST.sha256    {len(lines):>4} files, "
          f"{total / 1e6:.1f} MB")

    (out / "outputs" / "status.json").write_text(
        json.dumps({"checks": statuses, "all_passed": not failed}, indent=2),
        encoding="utf-8")

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"  wrote {archive} "
              f"({Path(archive).stat().st_size / 1e6:.1f} MB)")

    print()
    if failed:
        print(f"  {len(failed)} check(s) FAILED: {', '.join(failed)}")
        print("  The package is built but must not be sent while a check "
              "fails.")
        return 1
    print("  Every check passed inside the package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

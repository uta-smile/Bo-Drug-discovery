"""The three things to do with a free history, on the 18 cross-family blocks.

    python scripts/run_xfam_options.py            # run what is missing
    python scripts/run_xfam_options.py --dry-run  # show the plan

FINDINGS 38 located a seeding threshold on these blocks and FINDINGS 42 showed
the headline multi-task gain holds under a uniform history. Neither says which
of the available uses of a free history a practitioner should pick, because the
two were never measured under one protocol.

This runs the comparison. Per block, three options with identical budget,
identical library and identical history draw:

* **abstain**  -- single-task BO, random initial design (the ``stl`` arm, which
                  is the baseline every cell already carries).
* **seed**     -- single-task BO, initial design drawn from the history's
                  best-scoring compounds (``--seed-pool aux_top``).
* **transfer** -- coregionalised GP on the history, random initial design.

and crossed with the history design, because FINDINGS 40 showed it is not
neutral in either setting:

* **uniform**  -- the history may contain the primary's own top-k. This is the
                  primary design (FINDINGS 42); a method cannot read the
                  protocol, but the seeding pool is enriched in the answer.
* **withheld** -- ``--history-excludes-top``. The seeding pool cannot contain
                  the answer, which is how FINDINGS 38 measured the threshold,
                  but a method that avoids the history is walking toward it.

Reporting both is the point. ``scripts/transfer_rule.py`` then fits a decision
rule on one set and spends the held-out blocks.

Output goes to ``results/xfam_{transfer,seed}[_excl]/kNN.json``. Existing files
are kept; delete one to re-run it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BLOCKS = ROOT / "results" / "cross_family_blocks.json"
SOURCE = Path("data/cache/tdc/kiba_score.csv")

# Matched to the blocks: they are 428-433 compounds, so 95 is ~22% of the
# library, the low end of the operating window. KIBA runs keep censored values
# (the filter leaves 25-99 of ~900 rows there, so applying it would measure a
# correlation no campaign sees).
BASE = ["--source", str(SOURCE), "--keep-censored", "--acquisition", "ei",
        "--budget", "95", "--batch", "10", "--n-init", "10", "--k", "10",
        "--history", "50", "--repeats", "32", "--seed", "0", "--audit"]

CONDITIONS = (
    # (directory, extra argv, what it measures)
    ("xfam_transfer",      ["--arms", "stl", "mt"],                     []),
    ("xfam_seed",          ["--arms", "stl", "--seed-pool", "aux_top"], []),
    ("xfam_transfer_excl", ["--arms", "stl", "mt"],
     ["--history-excludes-top"]),
    ("xfam_seed_excl",     ["--arms", "stl", "--seed-pool", "aux_top"],
     ["--history-excludes-top"]),
)


def plan(n_blocks):
    out = []
    for d, arms, design in CONDITIONS:
        for i in range(1, n_blocks + 1):
            label = f"k{i:02d}"
            path = ROOT / "results" / d / f"{label}.json"
            argv = [PY, str(ROOT / "scripts" / "dense_block_campaign.py"),
                    "--blocks-from", str(BLOCKS), "--family", label,
                    *BASE, *arms, *design, "--out", str(path)]
            out.append((argv, path, f"{d}/{label}"))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", type=int, default=18)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cells = plan(args.blocks)
    todo = [c for c in cells if not c[1].exists()]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} already present, "
          f"{len(todo)} to run")
    if args.dry_run:
        for _, path, label in cells:
            print(f"  {'have' if path.exists() else 'RUN '}  {label}")
        return 0

    t0 = time.time()
    for n, (cmd, path, label) in enumerate(todo, 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.with_suffix(".log")
        with log.open("w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and path.exists()
        print(f"  [{n}/{len(todo)}] {label}  "
              f"{'ok' if ok else 'FAILED, see ' + log.name}  "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        if not ok:
            return 1
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

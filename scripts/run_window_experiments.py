"""Every dense-block sweep in sections 34-35, as one table and one runner.

    python scripts/run_window_experiments.py --list
    python scripts/run_window_experiments.py window --dry-run
    python scripts/run_window_experiments.py acquisition oracle controls

This replaces six shell drivers that were 183 lines of the same
`dense_block_campaign.py` invocation with two or three fields swapped, each
carrying its own hardcoded copy of the post-censoring block sizes. Those copies
were a correctness hazard rather than a tidiness one: the sizes depend on
`--pile-frac`, nothing re-derived them, and they had already drifted -- JAK's
matched-coverage cell ran at budget 199 against its window cell at 200, which
quietly broke the pairing those two sweeps are compared on.

Here the sizes come from the data every time, through the same
`drop_censored` the campaign itself uses, so a cell cannot be built from a stale
number. What is left to read is one `EXPERIMENTS` table saying what each sweep
varies, which is the only thing that actually differs between them.

Artefact names are unchanged, so this re-runs nothing that already exists and
the analysis scripts keep globbing what they always did. `--dry-run` prints the
plan without spending anything; run it first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Common to every cell; only the EXPERIMENTS table below departs from these.
BASE = {"batch": 10, "n_init": 10, "k": 10, "history": 50, "repeats": 8}

# Section 25 ran JAK on round budgets rather than exact fractions of the block.
# Keeping them means every later sweep stays paired with those artefacts; the
# alternative is a 199-vs-200 mismatch, which is how the drift above happened.
BUDGET_OVERRIDE = {("jak", 11): 100, ("jak", 22): 200, ("jak", 33): 300,
                   ("jak", 50): 450, ("jak", 66): 600}

EXPERIMENTS = {
    "window": {
        "doc": "the operating window, 11/22/33/50% of library (section 34)",
        "families": ("pi3k", "hdac", "dpp", "mmp"),
        "fracs": (11, 22, 33, 50),
        "acquisitions": ("ei",),
        "arms": ("stl", "mt", "pooled"),
        "prefix": "win",
    },
    "power": {
        "doc": "the window cells at 32 repeats -- is a 3/4 real or underpowered?",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("ei",),
        "arms": ("stl", "mt", "pooled"),
        "repeats": 32,
        "prefix": "win32",
    },
    "coverage": {
        "doc": "matched history coverage -- H set per family so the free "
               "history covers 25% of every block",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("ei",),
        "arms": ("stl", "mt", "pooled"),
        "coverage": 0.25,
        "prefix": "wincov",
    },
    "leak": {
        "doc": "the window cells again with a UNIFORM history -- the other "
               "side of the --history-excludes-top control (section 40)",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("ei",),
        "arms": ("stl", "mt", "pooled"),
        "repeats": 32,
        "excludes_top": False,
        "prefix": "winleak",
    },
    "leakcov": {
        "doc": "matched 25% coverage AND a uniform history -- separates the "
               "two things section 40 found confounded",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("ei",),
        "arms": ("stl", "mt", "pooled"),
        "repeats": 32,
        "coverage": 0.25,
        "excludes_top": False,
        "prefix": "leakcov",
    },
    "acquisition": {
        "doc": "the fixed acquisitions, against section 34's EI (section 35)",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("ucb", "pi", "greedy", "ts"),
        "arms": ("stl", "mt"),
        "prefix": "acq",
    },
    "oracle": {
        "doc": "the per-round hindsight ceiling over the portfolio. READS THE "
               "TRUE VALUES -- a reference line, not an arm",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("oracle",),
        "arms": ("stl", "mt"),
        "prefix": "acq",
    },
    "controls": {
        "doc": "the free switching schedules the ceiling has to be read against",
        "families": ("jak", "cyp", "pi3k", "hdac", "dpp", "mmp"),
        "fracs": (22, 33),
        "acquisitions": ("switch_rr", "switch_random"),
        "arms": ("stl", "mt"),
        "prefix": "acq",
    },
}


def clean_size(family, pile_frac=0.02):
    """Compounds surviving the censoring filter -- the denominator for a budget.

    Derived, never stored. The sizes depend on ``pile_frac``, and a copy of them
    in a driver script goes stale silently the moment that changes.
    """
    from dense_block_campaign import FAMILIES
    from dense_block_h1 import build_block, drop_censored

    mapping = FAMILIES[family]
    block, _ = build_block(mapping)
    targets = [t for t in mapping.values() if t in block.columns]
    clean, _ = drop_censored(block, targets, pile_frac=pile_frac, verbose=False)
    return len(clean), len(targets)


def plan(name, spec, pile_frac=0.02):
    """Every cell the experiment would run, as (argv, output path)."""
    out = []
    for family in spec["families"]:
        n, n_targets = clean_size(family, pile_frac)
        # Matched-coverage sweeps set H from the block; everything else uses the
        # fixed H=50 the rest of the package is stated in.
        history = (int(spec["coverage"] * (n - BASE["k"]) / (n_targets - 1))
                   if "coverage" in spec else spec.get("history", BASE["history"]))
        for frac in spec["fracs"]:
            budget = BUDGET_OVERRIDE.get((family, frac), round(n * frac / 100))
            for acq in spec["acquisitions"]:
                # `acq_*` artefacts carry the acquisition in the name because
                # the sweep varies it; the others do not because it is fixed.
                stem = (f"{spec['prefix']}_{family}_b{budget}_{acq}"
                        if spec["prefix"] == "acq"
                        else f"{spec['prefix']}_{family}_b{budget}")
                path = ROOT / "results" / f"{stem}.json"
                argv = [PY, str(ROOT / "scripts" / "dense_block_campaign.py"),
                        "--family", family,
                        "--history", str(history),
                        "--budget", str(budget),
                        "--batch", str(BASE["batch"]),
                        "--n-init", str(BASE["n_init"]),
                        "--k", str(BASE["k"]),
                        "--acquisition", acq,
                        "--repeats", str(spec.get("repeats", BASE["repeats"])),
                        "--pile-frac", str(pile_frac),
                        # Not neutral in either setting -- see FINDINGS 40.
                        # True everywhere else so every stored cell replans
                        # byte for byte.
                        *(["--history-excludes-top"]
                          if spec.get("excludes_top", True) else []),
                        "--audit",
                        "--arms", *spec["arms"],
                        "--out", str(path)]
                out.append((argv, path, f"{family} {frac}% b{budget} H={history} {acq}"))
    # Cheapest first, so a long sweep returns something early.
    return sorted(out, key=lambda c: int(c[0][c[0].index("--budget") + 1]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Validated by hand rather than with `choices`: allowing zero arguments
    # there needs `[*EXPERIMENTS, []]`, which makes argparse advertise `[]`
    # as a selectable experiment.
    p.add_argument("experiments", nargs="*", metavar="EXPERIMENT",
                   help="one or more of: " + ", ".join(EXPERIMENTS))
    p.add_argument("--list", action="store_true", help="describe each and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and what is already present")
    p.add_argument("--pile-frac", type=float, default=0.02)
    args = p.parse_args(argv)

    unknown = [e for e in args.experiments if e not in EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiment{'s' if len(unknown) > 1 else ''} "
                f"{', '.join(map(repr, unknown))}; choose from "
                + ", ".join(EXPERIMENTS))

    if args.list or not args.experiments:
        width = max(len(k) for k in EXPERIMENTS)
        for name, spec in EXPERIMENTS.items():
            n = len(spec["families"]) * len(spec["fracs"]) * len(spec["acquisitions"])
            print(f"  {name:<{width}}  {n:>2} cells  {spec['doc']}")
        if not args.experiments:
            print("\nnothing selected; pass one or more names, or --list")
        return 0

    todo, have = [], 0
    for name in args.experiments:
        cells = plan(name, EXPERIMENTS[name], args.pile_frac)
        print(f"\n{name}: {EXPERIMENTS[name]['doc']}")
        for argv_, path, label in cells:
            if path.exists():
                have += 1
                if args.dry_run:
                    print(f"    have  {label:<34} {path.name}")
            else:
                todo.append((argv_, path, label))
                print(f"    {'would run' if args.dry_run else 'run':<5} "
                      f"{label:<34} {path.name}")

    print(f"\n{have} already present, {len(todo)} to run")
    if args.dry_run or not todo:
        return 0
    for i, (argv_, path, label) in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {label} ===", flush=True)
        if subprocess.run(argv_, cwd=ROOT).returncode:
            print(f"FAILED {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

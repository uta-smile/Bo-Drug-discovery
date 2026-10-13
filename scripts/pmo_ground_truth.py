"""Score every molecule in the pool, so a PMO task's true optimum is known.

    python scripts/pmo_ground_truth.py --tasks albuterol_similarity qed

A PMO task is a scoring function from SMILES to [0, 1] that costs microseconds,
so "what is the best answer on this library?" is answerable by brute force:
evaluate all 249,455 ZINC-250k molecules and sort. That is not available for a
real assay, which is the whole reason Bayesian optimisation exists, but it is
available here -- and it converts every AUC in the PMO tier from a number that
can only be compared against other methods into one that can be compared
against the answer.

Two things the output is, and one it is not
------------------------------------------
``pool_top10_mean``
    The mean of the ten highest scores in the pool. This is the **ceiling on
    the AUC top-10 metric**: the metric is the average of the top-10 curve, the
    curve cannot exceed this value, so no method can either. A method that
    identified the ten best molecules on its first ten calls would score
    exactly this.
``pool_top1``
    The single best score in the pool -- the best any method can report as its
    top hit.

It is **not** the global optimum of the task. The search space here is a fixed
library, and these tasks are defined over all of chemical space. For the
rediscovery tasks especially the difference is large: ``celecoxib_rediscovery``
scores 1.0 for celecoxib itself, but celecoxib is not in ZINC-250k, so the best
the library admits is 0.4583. Generative methods that build molecules are not
bounded by the numbers here; library methods are.

Results accumulate in ``results/pmo_ground_truth.json``. A task already present
is skipped unless ``--force`` is passed, since the pool and the oracle are both
fixed and the scan is the expensive part (90 seconds for a similarity task,
23 minutes for ``deco_hop``, which does SMARTS matching on every molecule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.oracles.pmo import PMO_TASKS, PMOOracle, probe_oracle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZINC_TAB = ROOT / "data" / "cache" / "tdc" / "zinc.tab"
OUT = ROOT / "results" / "pmo_ground_truth.json"
CHUNK = 20_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tasks", nargs="+", default=list(PMO_TASKS))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--force", action="store_true", help="rescan tasks already in the file")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def load_pool() -> list[str]:
    import pandas as pd

    if not ZINC_TAB.exists():
        print(
            f"error: {ZINC_TAB} not found. Fetch it once with:\n"
            "  python -c \"from tdc.generation import MolGen; "
            "MolGen(name='ZINC', path='data/cache/tdc')\"",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return pd.read_csv(ZINC_TAB, sep="\t")["smiles"].tolist()


def scan(task: str, smiles: list[str], progress: bool = True) -> dict:
    """Exhaustively score ``smiles`` and summarise the top of the distribution.

    The oracle is built with ``budget=None`` and called through its TDC handle
    rather than through ``__call__``: this is a census of the library, not a
    campaign, and it must not pass through the budget accounting that every
    real run is measured by.
    """
    oracle = PMOOracle(task, budget=None)
    t0 = time.perf_counter()
    vals: list[float] = []
    for i in range(0, len(smiles), CHUNK):
        raw = oracle._tdc(smiles[i : i + CHUNK])
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        vals.extend(float(v) for v in raw)
        if progress:
            done = min(i + CHUNK, len(smiles))
            print(f"  {task} {done:,}/{len(smiles):,}  {time.perf_counter() - t0:.0f}s",
                  flush=True)

    s = np.sort(np.asarray(vals, dtype=float))
    return {
        "pool_top1": float(s[-1]),
        "pool_top10_mean": float(s[-10:].mean()),
        "pool_top100_mean": float(s[-100:].mean()),
        "pool_mean": float(s.mean()),
        "n_scored": int(s.size),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    unknown = set(args.tasks) - set(PMO_TASKS)
    if unknown:
        print(f"error: unknown task(s) {sorted(unknown)}", file=sys.stderr)
        return 2

    out = Path(args.out)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    doc.setdefault("pool", "ZINC-250k")
    doc.setdefault("tasks", {})

    todo = [t for t in args.tasks if args.force or t not in doc["tasks"]]
    if not todo:
        print(f"nothing to do; {len(doc['tasks'])} task(s) already in {out}")
        return 0

    smiles = load_pool()
    doc["pool_size"] = len(smiles)
    print(f"pool: {len(smiles):,} molecules -> {len(todo)} task(s) to scan", flush=True)

    for task in todo:
        # A broken oracle scores every molecule identically; scanning 250k of
        # them to discover that wastes minutes and writes a ceiling of 0.0 that
        # later reads as a real result.
        status, detail = probe_oracle(task)
        if status == "broken":
            print(f"skip {task}: {detail}", file=sys.stderr)
            doc["tasks"][task] = {"skipped": detail}
            continue

        row = scan(task, smiles, progress=not args.quiet)
        row["probe"] = status
        doc["tasks"][task] = row
        print(f"{task}: top-1 {row['pool_top1']:.4f}  top-10 mean "
              f"{row['pool_top10_mean']:.4f}  ({row['wall_s']:.0f}s)", flush=True)
        # Written after every task: the long scans are worth checkpointing.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    print(f"\nwrote {out} ({len(doc['tasks'])} task(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

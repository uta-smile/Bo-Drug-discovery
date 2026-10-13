"""Record plain-BO shortlists, labelled with their true values, as CLM training data.

    python scripts/clm_collect.py \
        --datasets data/chembl/ache_all.csv data/chembl/vegfr2_all.csv \
                   data/chembl/jak2_all.csv data/chembl/cdk2_all.csv data/chembl/pi3ka_all.csv \
        --seeds 10 --budget 200 --out data/clm/train.jsonl

    python scripts/clm_collect.py --datasets data/chembl/egfr_anilinoquinazoline.csv \
        --seeds 5 --seed-start 1000 --out data/clm/egfr_test.jsonl

Each run is plain BO (``bo_ei``) with a :class:`~molbo.clm.RecordingSelector`
that always takes rank 0, so the trajectory is exactly plain BO and every
recorded shortlist is one the real selector would have faced. One JSON line per
round: the observations so far, the EI top-k shortlist, its posterior and its
true values.

Keep EGFR out of the training file. It is the lookup tier's test series; the
second command above records it separately, from seeds the sweep does not use,
for offline evaluation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molbo.campaign import ArmConfig  # noqa: E402
from molbo.clm import RecordingSelector  # noqa: E402
from molbo.experiment import run_sweep  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--shortlist-k", type=int, default=15)
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--min-heavy-atoms", type=int, default=10, help="same filter as the sweep")
    p.add_argument("--max-heavy-atoms", type=int, default=55)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for path in args.datasets:
            oracle = LookupOracle.from_dataset(path, value_column=args.value_column,
                                               budget=args.budget,
                                               min_heavy_atoms=args.min_heavy_atoms,
                                               max_heavy_atoms=args.max_heavy_atoms)
            rec = RecordingSelector(values=dict(oracle._table), tag=Path(path).stem)
            arm = ArmConfig("bo_ei_record", strategy="bo", acquisition="ei",
                            n_init=args.n_init, selector=rec, shortlist_k=args.shortlist_k)
            print(f"[collect] {rec.tag}: {len(oracle.library)} compounds, {args.seeds} seeds", flush=True)
            run_sweep(oracle, [arm], range(args.seed_start, args.seed_start + args.seeds),
                      budget=args.budget, verbose=False)
            for r in rec.records:
                f.write(json.dumps(r) + "\n")
            n += len(rec.records)
            print(f"[collect] {rec.tag}: {len(rec.records)} shortlists", flush=True)
    print(f"[collect] wrote {n} shortlists -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pull the two public kinome profiling panels into the dump format.

    python scripts/fetch_kinase_panels.py

Section 36's gate found no co-measured auxiliary outside a target's own paralogue
group anywhere in BindingDB. That is a fact about how medicinal-chemistry data is
*generated*: a team makes fifty analogues, runs them against its own target and
the three paralogues it needs selectivity over, and publishes. The blocks come
out family-shaped because the experiments were family-shaped.

Kinome profiling panels are generated the other way round -- one fixed compound
set pushed across a large slice of the kinome -- so if cross-family
co-measurement exists anywhere, it is here. Two are public:

* **DAVIS** (Davis et al. 2011) -- 68 compounds x 379 kinases of Kd, and the
  matrix is *completely* observed: 25,772 rows is exactly 68 x 379. It is the
  cleanest geometry in the field and it is still unusable for this protocol,
  because the protocol needs 160 *compounds* and this has 68. Fetched anyway, so
  that statement is measured rather than asserted.
* **KIBA** (Tang et al. 2014) -- 2,068 compounds x 229 kinases, ~25% observed.
  Wide enough on the compound axis to clear the floor, which makes it the one
  real candidate.

Both are written in the five-column shape of the BindingDB dump
(``ID1,X1,ID2,X2,Y``) so `find_dense_blocks` and `auxiliary_candidates` read them
through ``--source`` with no other change.

**Y does not mean the same thing across these files** and nothing downstream
should assume it does. BindingDB's is IC50 in nM and KIBA's is an integrated
score on its own scale, so the pIC50 conversion in ``dense_block_h1.build_block``
is wrong for KIBA. The co-measurement question only reads *which cells are
filled*, which is why it is answerable here and the censoring filter is not.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

PANELS = {"DAVIS": "davis_kd.csv", "KIBA": "kiba_score.csv"}
OUT_DIR = Path("data/cache/tdc")


def load_tdc_dti():
    """`tdc.multi_pred` imports optional single-cell deps at package import.

    tiledbsoma, cellxgene_census and gget are pulled in by `perturboutcome`,
    which has nothing to do with drug-target interaction data. Installing three
    heavy packages to reach a class that does not use them is the wrong trade, so
    stub whatever the import asks for and stop if it asks for something
    unexpected rather than looping.
    """
    stubbed = []
    for _ in range(10):
        try:
            from tdc.multi_pred import DTI
            return DTI, stubbed
        except ModuleNotFoundError as e:
            sys.modules[e.name] = types.ModuleType(e.name)
            stubbed.append(e.name)
    raise SystemExit("tdc.multi_pred still unimportable after stubbing "
                     + ", ".join(stubbed))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", nargs="*", default=list(PANELS), choices=list(PANELS))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)

    DTI, stubbed = load_tdc_dti()
    if stubbed:
        print(f"stubbed optional TDC deps: {', '.join(stubbed)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.panels:
        df = DTI(name=name, path=str(args.out_dir)).get_data()
        df = df.rename(columns={"Drug_ID": "ID1", "Drug": "X1",
                                "Target_ID": "ID2", "Target": "X2"})
        dest = args.out_dir / PANELS[name]
        df[["ID1", "X1", "ID2", "X2", "Y"]].to_csv(dest, index=False)
        n_c, n_t = df.X1.nunique(), df.ID2.nunique()
        print(f"{name:<6} {len(df):>8,} rows  {n_c:>5,} compounds x {n_t:>4} "
              f"targets  {len(df) / (n_c * n_t):>6.1%} observed  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

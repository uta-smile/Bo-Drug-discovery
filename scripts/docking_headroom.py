"""Does docking carry signal the optimiser does not already have?

    python scripts/docking_headroom.py \
        --dataset data/chembl/egfr_anilinoquinazoline.csv \
        --docking data/cache/docking/egfr_1m17.jsonl \
        --shortlists data/clm/egfr_test.jsonl

The gate on the whole external-tool layer. If a Vina score adds nothing to what
a Tanimoto GP on fingerprints already believes, then there is no information
for a tool-selecting policy to allocate, and the agentic framing collapses --
for the cost of an afternoon rather than a trained model.

Three questions, cheapest first:

1. **Does docking know anything at all?** Spearman against measured pIC50 over
   the whole library.
2. **Does it know anything *new*?** ``molbo.docking.incremental``: the change in
   R-squared from adding the docking score to the GP's posterior mean, and the
   partial correlation with what the mean leaves unexplained. This is the one
   that matters. Docking that merely agrees with the surrogate is free to
   compute and worth nothing.
3. **Would it change a decision?** ``argmax(docking)`` scored as a selector on
   recorded shortlists, against the same references
   ``scripts/selector_headroom.py`` uses.

Run ``scripts/dock_library.py`` first; it writes the cache this reads and
refuses to produce one whose redocking check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.clm import shortlist_metrics  # noqa: E402
from molbo.docking import DockingScores, incremental  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--docking", required=True, help=".jsonl from dock_library.py")
    p.add_argument("--shortlists", required=True, help=".jsonl from clm_collect.py")
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--smiles-column", default="canonical_smiles")
    p.add_argument("--out", default="results/docking_headroom.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from scipy.stats import spearmanr

    args = parse_args(argv)
    dock = DockingScores.load(args.docking)

    import pandas as pd

    df = pd.read_csv(args.dataset)
    smiles = list(df[args.smiles_column])
    truth = np.array(df[args.value_column], dtype=float)
    d = dock.values(smiles)
    ok = np.isfinite(d)
    print(f"[dock-headroom] library {len(smiles)} compounds, "
          f"{ok.sum()} with a Vina score ({100 * ok.mean():.1f}% coverage)")
    if (redock := dock.meta.get("redock")):
        print(f"[dock-headroom] receptor {dock.meta.get('pdb')} redock: "
              f"nn-RMSD {redock['nn_rmsd']:.2f} A, "
              f"{redock['within_2A']}/{redock['n_atoms']} atoms within 2 A")
    if ok.sum() < 50:
        raise SystemExit("too few scored molecules; run scripts/dock_library.py first")

    lib_rho = float(spearmanr(d[ok], truth[ok]).statistic)
    print(f"\n[1] library-wide  spearman(docking, pIC50) = {lib_rho:+.3f}   n={int(ok.sum())}")
    print("    For reference, a fingerprint ridge reaches 0.64-0.74 on this kind of split.")

    recs = [json.loads(l) for l in open(args.shortlists, encoding="utf-8") if l.strip()]
    flat_d, flat_mu, flat_true = [], [], []
    for r in recs:
        flat_d.extend(dock.values(r["shortlist"]))
        flat_mu.extend(r["mu"])
        flat_true.extend(r["true"])
    inc = incremental(np.array(flat_d), np.array(flat_mu), np.array(flat_true))
    print(f"\n[2] on {len(recs)} recorded shortlists ({inc['n']} scored candidates)")
    print(f"    spearman(docking, true pIC50)      {inc['rho_dock_true']:+.3f}")
    print(f"    spearman(docking, GP posterior mu) {inc['rho_dock_mu']:+.3f}"
          "   <- high means it just agrees with the surrogate")
    print(f"    R2 from mu alone                   {inc['r2_mu']:.3f}")
    print(f"    R2 from mu + docking               {inc['r2_mu_plus_dock']:.3f}"
          f"   (delta {inc['delta_r2']:+.3f})")
    print(f"    partial spearman vs mu's residual  {inc['partial_rho']:+.3f}"
          "   <- the signal that is genuinely new")

    true = [np.asarray(r["true"]) for r in recs]
    rows = {
        "docking": shortlist_metrics([dock.values(r["shortlist"]) for r in recs], true),
        "max_mu": shortlist_metrics([np.asarray(r["mu"]) for r in recs], true),
        "bo_rank0": shortlist_metrics([-np.arange(len(r["true"])) for r in recs], true),
        "oracle": shortlist_metrics(true, true),
    }
    print(f"\n[3] as a selector on those shortlists (gain: 0 = random pick)")
    print(f"    {'selector':<12s}{'gain':>10s}{'se':>8s}{'regret':>9s}{'top1':>8s}")
    for name, m in rows.items():
        print(f"    {name:<12s}{m['gain']:>+10.3f}{m['gain_se']:>8.3f}"
              f"{m['regret']:>9.3f}{m['top1']:>8.3f}")

    dock_gain, mu_gain = rows["docking"]["gain"], rows["max_mu"]["gain"]
    se = rows["docking"]["gain_se"]
    print("\nVERDICT")
    if abs(inc["partial_rho"]) < 0.05 and dock_gain < 2 * se:
        print("  Docking adds nothing the GP does not already have, and does not beat a")
        print("  random pick. There is no information for a tool-selecting policy to")
        print("  allocate. Retire the external-tool layer -- this is the cheap negative")
        print("  the instrument exists to produce.")
    elif inc["partial_rho"] >= 0.05 and dock_gain < mu_gain:
        print(f"  Docking carries genuinely new signal (partial rho {inc['partial_rho']:+.3f})")
        print(f"  but is a worse selector on its own ({dock_gain:+.3f} vs max_mu's {mu_gain:+.3f}).")
        print("  That combination is exactly the case for BLENDING rather than switching:")
        print("  run DockSelector with --blend-mu, and keep docking as a feature, not a rule.")
    else:
        print(f"  Docking beats the surrogate's own pick ({dock_gain:+.3f} vs {mu_gain:+.3f})")
        print(f"  and adds new signal (partial rho {inc['partial_rho']:+.3f}). Worth a campaign")
        print("  arm at 200 seeds, and worth asking a policy when to spend on it.")
    print("\n  Either way, remember the ceiling: the whole shortlist hook is worth")
    print(f"  +0.031 AUC, and perfect hindsight selection reaches gain "
          f"{rows['oracle']['gain']:+.3f} here.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.dataset, "docking": args.docking, "shortlists": args.shortlists,
        "library_size": len(smiles), "coverage": float(ok.mean()),
        "library_spearman": lib_rho, "incremental": inc,
        "selectors": rows, "redock": dock.meta.get("redock"),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

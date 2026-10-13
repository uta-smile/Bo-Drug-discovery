"""Which molecular representation actually carries potency signal?

    python scripts/clm_probe.py --train data/clm/train.jsonl --within cdk2_all
    python scripts/clm_probe.py --train data/clm/train.jsonl --within cdk2_all \
        --reps fingerprint descriptors chemberta-mtr molformer

The gate on everything downstream. A contrastive head reshapes whatever the
encoder hands it; it cannot recover a signal the representation never had. So
before training any head, fit the cheapest possible model -- ridge regression
-- from each candidate representation to measured potency, and see which ones
know anything at all.

The split is **within one target, by Murcko scaffold**. Within-target because
that is the regime a campaign is in: some of a series measured, the rest
predicted. By scaffold because a random split leaves near-duplicates on both
sides and every representation then looks excellent.

Two references decide the outcome:

``fingerprint``
    ECFP4, what the GP surrogate already uses. A learned representation that
    loses to this is contributing nothing the campaign does not have for free,
    and it costs a GPU to produce.
``scaffold``
    Predict each compound by its Murcko scaffold's training mean. A congeneric
    series is largely predictable from its scaffold, so this is the floor.

Measured on the first pass: Qwen3.5-4B last-token scored Spearman 0.21-0.27
against the fingerprint's 0.64-0.73 -- a 4.7-billion-parameter model reading
molecules worse than 2,048 bits. That result is what this bake-off exists to
act on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.clm import BACKBONE, BACKBONE_REVISION, Encoder  # noqa: E402

ENCODERS: dict[str, dict] = {
    # The v1 backbone, both poolings. `last` is what produced the failed run;
    # `mean` is the cheapest hypothesis for why -- last-token pooling on a
    # *base* (non-instruction-tuned) model is a known-weak sentence embedding.
    "qwen-last": dict(model=BACKBONE, revision=BACKBONE_REVISION, pooling="last", max_len=2048),
    "qwen-mean": dict(model=BACKBONE, revision=BACKBONE_REVISION, pooling="mean", max_len=2048),
    # Chemistry-pretrained, and three orders of magnitude smaller. MTR was
    # trained on regression targets rather than masked tokens, so it is the
    # likelier of the two ChemBERTas to carry property signal.
    "chemberta-mlm": dict(model="DeepChem/ChemBERTa-77M-MLM", revision=None,
                          pooling="mean", max_len=512),
    "chemberta-mtr": dict(model="DeepChem/ChemBERTa-77M-MTR", revision=None,
                          pooling="mean", max_len=512),
    "chemberta-zinc": dict(model="seyonec/ChemBERTa-zinc-base-v1", revision=None,
                           pooling="mean", max_len=512),
    "molformer": dict(model="ibm-research/MoLFormer-XL-both-10pct", revision=None,
                      pooling="mean", max_len=202, trust_remote_code=True),
}

NON_NEURAL = ("fingerprint", "descriptors")
DEFAULT_REPS = ("fingerprint", "descriptors", "chemberta-mtr", "molformer", "qwen-last")


def molecules(path: str, keep=None, drop=None) -> dict[str, float]:
    """Unique ``smiles -> pIC50`` over every shortlist in a collection file.

    ``keep`` / ``drop`` filter on the record's ``dataset``, so one collection
    file can supply both sides of a target-held-out split.
    """
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (keep and r["dataset"] != keep) or (drop and r["dataset"] == drop):
                continue
            out.update(zip(r["shortlist"], r["true"]))
    return out


def ridge(Xtr, ytr, Xte, alpha: float):
    """Closed-form ridge on centred features; returns predictions for ``Xte``."""
    mu, ym = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return (Xte - mu) @ w + ym


def scores(y, pred) -> dict:
    from scipy.stats import spearmanr

    ss = float(((y - y.mean()) ** 2).sum())
    return {"r2": 1.0 - float(((y - pred) ** 2).sum()) / ss,
            "rho": float(spearmanr(y, pred).statistic),
            "mae": float(np.abs(y - pred).mean())}


def featurise(rep: str, smiles: list[str], cache_dir: str) -> tuple[np.ndarray, str]:
    """-> ``([n, d]`` matrix, a human label for the table)."""
    if rep == "fingerprint":
        from molbo.surrogate import fingerprint_matrix

        return fingerprint_matrix(smiles).astype(np.float64), "ECFP4 2048-bit"
    if rep == "descriptors":
        from molbo.constraints import descriptor_table

        table = descriptor_table(smiles)
        X = np.column_stack([table[k] for k in sorted(table)])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Descriptors span wildly different scales (MW ~500, fsp3 ~0.3); one
        # ridge penalty over raw columns would regularise them incomparably.
        sd = X.std(0)
        sd[sd == 0] = 1.0
        return (X - X.mean(0)) / sd, f"RDKit x{X.shape[1]}"
    spec = ENCODERS[rep]
    enc = Encoder(cache_dir=cache_dir, **spec)
    return enc.embed(smiles, progress=True).astype(np.float64), spec["model"].split("/")[-1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", help="separate collection file for the held-out target")
    p.add_argument("--holdout", help="instead of --test, hold out this dataset from --train")
    p.add_argument("--within", help="fit and test inside one dataset, split by Murcko "
                                    "scaffold. This is the regime the campaign is in")
    p.add_argument("--reps", nargs="+", default=list(DEFAULT_REPS),
                   help=f"any of: {', '.join(NON_NEURAL)}, {', '.join(ENCODERS)}")
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--split-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="data/cache/clm")
    p.add_argument("--out", default=None, help="write the table as JSON")
    args = p.parse_args(argv)

    unknown = [r for r in args.reps if r not in ENCODERS and r not in NON_NEURAL]
    if unknown:
        p.error(f"unknown representation(s) {unknown}")
    if not (args.test or args.holdout or args.within):
        p.error("pass --test, --holdout or --within")

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")

    def scaffold(s: str) -> str:
        m = Chem.MolFromSmiles(s)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else s

    if args.within:
        every = molecules(args.train, keep=args.within)
        if not every:
            p.error(f"--within {args.within!r} matched no records in {args.train}")
        groups: dict[str, list[str]] = {}
        for s in sorted(every):
            groups.setdefault(scaffold(s), []).append(s)
        order = sorted(groups)
        np.random.default_rng(args.seed).shuffle(order)
        cut = int(round(args.split_frac * len(every)))
        held: set[str] = set()
        for g in order:
            if len(held) >= cut:
                break
            held.update(groups[g])
        tr = {s: v for s, v in every.items() if s not in held}
        te = {s: v for s, v in every.items() if s in held}
        label = f"{args.within} (within, scaffold split, {len(groups)} scaffolds)"
    else:
        tr = molecules(args.train, drop=args.holdout)
        te = (molecules(args.train, keep=args.holdout) if args.holdout
              else molecules(args.test))
        shared = set(tr) & set(te)
        for s in shared:
            del te[s]
        label = (f"holdout {args.holdout}" if args.holdout else args.test) + \
                f" ({len(shared)} shared molecules dropped from test)"

    tr_s, te_s = sorted(tr), sorted(te)
    ytr = np.array([tr[s] for s in tr_s])
    yte = np.array([te[s] for s in te_s])
    print(f"[probe] {label}\n[probe] train {len(tr_s)} molecules, test {len(te_s)}\n")

    rows = []
    print(f"{'representation':<18s}{'what it is':<22s}{'dim':>6s}{'R2':>9s}{'spearman':>10s}{'MAE':>7s}{'sec':>7s}")
    print("-" * 79)
    for rep in args.reps:
        t0 = time.time()
        try:
            X, what = featurise(rep, tr_s + te_s, args.cache_dir)
        except Exception as exc:  # a missing model must not lose the other rows
            print(f"{rep:<18s}FAILED  {type(exc).__name__}: {str(exc)[:44]}")
            rows.append({"representation": rep, "error": f"{type(exc).__name__}: {exc}"})
            continue
        dt = time.time() - t0
        m = scores(yte, ridge(X[:len(tr_s)], ytr, X[len(tr_s):], args.alpha))
        rows.append({"representation": rep, "what": what, "dim": int(X.shape[1]),
                     "seconds": dt, **m})
        print(f"{rep:<18s}{what:<22s}{X.shape[1]:>6d}{m['r2']:>9.3f}{m['rho']:>10.3f}"
              f"{m['mae']:>7.3f}{dt:>7.1f}")

    by = {s: [] for s in {scaffold(s) for s in tr_s}}
    for s, v in zip(tr_s, ytr):
        by[scaffold(s)].append(v)
    means = {k: float(np.mean(v)) for k, v in by.items()}
    m = scores(yte, np.array([means.get(scaffold(s), ytr.mean()) for s in te_s]))
    rows.append({"representation": "scaffold", "what": "training mean", **m})
    print(f"{'scaffold':<18s}{'training mean':<22s}{'-':>6s}{m['r2']:>9.3f}{m['rho']:>10.3f}{m['mae']:>7.3f}")
    print(f"{'global mean':<18s}{'':<22s}{'-':>6s}{0.0:>9.3f}{'-':>10s}"
          f"{float(np.abs(yte - ytr.mean()).mean()):>7.3f}")

    ok = [r for r in rows if "rho" in r and r["representation"] != "scaffold"]
    if ok:
        best = max(ok, key=lambda r: r["rho"])
        fp = next((r for r in ok if r["representation"] == "fingerprint"), None)
        print(f"\nVERDICT: best is {best['representation']} (spearman {best['rho']:.3f}).")
        if fp and best["representation"] != "fingerprint":
            print(f"  It beats the fingerprint by {best['rho'] - fp['rho']:+.3f} spearman. "
                  f"Worth the encoder only if that margin survives on other targets.")
        elif fp:
            print("  The fingerprint wins. Use it: no GPU, no cache, no embedding pass,\n"
                  "  and it makes scaling to many targets free.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": label, "n_train": len(tr_s), "n_test": len(te_s),
                                   "alpha": args.alpha, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does Boltz-2 know enough about potency to be worth its 20 s/ligand?

Memory, measured on a 16 GB RTX 5070 Ti with a 309-residue target and the
``--no_kernels`` fallback that Windows requires (the fused triangular-mult
kernels ship Linux wheels only, and they are what normally keeps this small):

    full MSA, 5 parallel affinity samples   15.9 GB   the boltz defaults
    full MSA, 1 parallel sample             13.4 GB
    full MSA, capped at 10 GB               OOM       every ligand fails
    MSA 2048/4096, capped at 11 GB           8.4 GB   <- the defaults here
    MSA 512/2048,  capped at 12 GB           6.2 GB

Install ``triton-windows`` into the boltz environment. Without it
``torch.utils._triton.has_triton()`` is False and every ``torch.compile`` path
silently runs eager; with it the same configuration goes 127 -> 111 s/ligand
and 8.4 -> 8.0 GB. It does *not* restore the fused triangular-multiplication
kernels -- those come from ``cuequivariance-ops-torch-cu12``, which publishes
Linux wheels only, and boltz marks that entry point ``@torch.compiler.disable``
so Triton cannot substitute. The gain is on the triangle-attention path.

Because a compiled kernel and its eager equivalent are not bit-identical,
changing whether Triton is installed is an implementation change like
``no_kernels``: clear the value cache rather than blending the two.

The defaults are the fourth row: the deepest alignment that fits comfortably
under 12 GB. Raising ``--num-subsampled-msa`` changes the prediction, not just
the runtime, which is why it is in the oracle cache key -- values computed at
two depths are two different numbers and must not share a results table.

    python scripts/boltz2_headroom.py --target O60674 --column JAK2 --n 120

This is the gate on the whole compute-oracle tier, and it is deliberately the
cheapest question in it. Sections 8 and 32 established two thresholds that
together decide whether a structure-based predictor is worth running at all:

* **A proxy used as the GP's prior mean must reach Spearman ~0.37** against the
  true objective before it pays (FINDINGS 8); below that it makes low-budget BO
  *worse*, and a zero-quality prior costs -0.049 AUC.
* **A proxy used as an auxiliary task must reach ~0.45-0.6** (FINDINGS 32,
  provisional -- measured under EI against a contaminated baseline).

Docking sits at 0.2-0.4 and clears neither, which is why this project never
built a docking cache. Boltz-2's affinity head is a meaningful step past
docking, so the question is whether it clears them -- and that is answerable
with N forward passes and no campaign at all.

**Correlation is necessary, not sufficient.** A predictor that merely agrees
with what a Tanimoto GP already believes is free to compute and worth nothing,
so the incremental quantity -- what Boltz-2 explains of what the fingerprint GP
gets *wrong* -- is reported beside the raw rho. Same decomposition
``scripts/docking_headroom.py`` uses, for the same reason.

Ligands come from a measured dense block, so every one has a real pIC50 to
score against. Nothing is being selected here: the sample is stratified across
the potency range rather than concentrated at the top, because a correlation
measured only on potent compounds is not the one a campaign would rely on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.metrics import spearman0  # noqa: E402
from molbo.oracles.boltz2 import Boltz2Config, Boltz2Oracle  # noqa: E402
from molbo.surrogate import (  # noqa: E402
    TanimotoGP,
    fingerprint_matrix,
    tanimoto_gram,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_block_campaign import FAMILIES  # noqa: E402
from dense_block_h1 import build_block, drop_censored  # noqa: E402

PRIOR_MEAN_THRESHOLD = 0.37
"""FINDINGS 8: below this a proxy used as the GP's prior mean costs AUC."""

AUX_TASK_THRESHOLD = 0.45
"""FINDINGS 32, low end and provisional: below this a proxy auxiliary does not
separate from a pure-noise one."""

SETUP_HINT = ("uv venv .venv-boltz --python 3.12 && "
              "VIRTUAL_ENV=.venv-boltz uv pip install boltz torch "
              "--torch-backend=cu128")


def stratified_sample(y: np.ndarray, n: int, rng) -> np.ndarray:
    """``n`` compounds spanning the potency range, not concentrated at the top.

    A sample of the most potent compounds would measure Boltz-2 over a range it
    will never be asked to rank in deployment, and would inflate or deflate the
    correlation depending on where its errors happen to sit. Equal counts per
    decile of the measured value keep the comparison over the whole range.
    """
    order = np.argsort(y)
    bins = np.array_split(order, 10)
    per = max(1, n // len(bins))
    picks = [b[rng.choice(len(b), size=min(per, len(b)), replace=False)]
             for b in bins]
    out = np.concatenate(picks)
    if len(out) > n:
        out = out[rng.choice(len(out), size=n, replace=False)]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="jak", choices=sorted(FAMILIES))
    p.add_argument("--target", default="O60674",
                   help="UniProt accession; the sequence must be in data/targets/")
    p.add_argument("--column", default="JAK2",
                   help="which measured column of the block to score against")
    p.add_argument("--n", type=int, default=120, help="ligands to evaluate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boltz-python", default=".venv-boltz/Scripts/python.exe",
                   help="interpreter that has boltz installed. Kept separate "
                        "because boltz pins numpy<2 and this project's stored "
                        "numbers were produced on numpy 2")
    p.add_argument("--cache", default="data/cache/boltz2.sqlite")
    p.add_argument("--work-dir", default="data/cache/boltz2_work")
    p.add_argument("--msa", default=None, help="precomputed .a3m for the target")
    p.add_argument("--use-msa-server", action="store_true")
    p.add_argument("--vram-gb", type=float, default=11.0,
                   help="hard GPU memory ceiling for the prediction process. "
                        "Exceeding it fails that ligand rather than the run")
    p.add_argument("--max-parallel-samples", type=int, default=1,
                   help="affinity diffusion samples per forward pass. The "
                        "dominant VRAM term; boltz defaults to 5")
    p.add_argument("--max-msa-seqs", type=int, default=4096)
    p.add_argument("--subsample-msa", action="store_true", default=True,
                   help="trim the alignment. Changes the prediction, so it is "
                        "part of the oracle cache key")
    p.add_argument("--num-subsampled-msa", type=int, default=2048)
    p.add_argument("--no-kernels", action="store_true",
                   help="pure-PyTorch triangular multiplication. Required on "
                        "Windows: the fused kernels ship Linux wheels only")
    p.add_argument("--keep-censored", action="store_true")
    p.add_argument("--out", default="results/boltz2_headroom.json")
    args = p.parse_args(argv)

    seq_path = Path("data/targets") / f"{args.target}.txt"
    if not seq_path.exists():
        raise SystemExit(f"no sequence at {seq_path}; fetch it with "
                         f"molbo.targets.binding_domain('{args.target}')")
    sequence = seq_path.read_text(encoding="utf-8").strip()

    # subprocess does not search the working directory for an executable on
    # Windows, so a relative interpreter path fails with a bare "cannot find
    # the file specified" raised from deep inside the oracle.
    boltz_python = str(Path(args.boltz_python).resolve())
    if not Path(boltz_python).exists():
        raise SystemExit(f"no interpreter at {boltz_python}. Create it: {SETUP_HINT}")

    t0 = time.time()
    mapping = FAMILIES[args.family]
    block, _ = build_block(mapping)
    targets = [t for t in mapping.values() if t in block.columns]
    if not args.keep_censored:
        block, _ = drop_censored(block, targets)
    if args.column not in targets:
        raise SystemExit(f"--column {args.column} not in block {targets}")

    y = block[args.column].to_numpy(float)
    rng = np.random.default_rng(args.seed)
    pick = stratified_sample(y, args.n, rng)
    smiles = [block.smiles.iloc[int(i)] for i in pick]
    truth = y[pick]
    print(f"{args.family} block: {len(block):,} compounds; scoring {len(pick)} "
          f"against measured {args.column} "
          f"({truth.min():.2f}..{truth.max():.2f} pIC50)")
    print(f"target {args.target}: {len(sequence)} residues")

    cfg = Boltz2Config(protein_sequence=sequence, msa_path=args.msa,
                       use_msa_server=args.use_msa_server,
                       no_kernels=args.no_kernels,
                       vram_gb=args.vram_gb,
                       max_parallel_samples=args.max_parallel_samples,
                       max_msa_seqs=args.max_msa_seqs,
                       subsample_msa=args.subsample_msa,
                       num_subsampled_msa=args.num_subsampled_msa)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    oracle = Boltz2Oracle(cfg, work_dir=args.work_dir, cache_path=args.cache,
                          boltz_python=boltz_python)
    print(f"deterministic: {cfg.is_deterministic} (gamma_0={cfg.gamma_0})")

    pred = oracle(smiles)
    ok = np.array([v is not None for v in pred])
    print(f"\n{int(ok.sum())}/{len(pred)} ligands scored "
          f"({time.time() - t0:.0f}s, {oracle.n_calls} oracle calls)")
    if ok.sum() < 10:
        raise SystemExit("too few successful predictions to correlate")

    idx_ok = np.flatnonzero(ok)
    pv = np.asarray([float(pred[int(i)]) for i in idx_ok])
    tv = truth[idx_ok]
    rho = spearman0(pv, tv)

    # What a fingerprint GP already believes, out of fold. Boltz-2 that merely
    # agrees with it is free to compute and worth nothing.
    X = fingerprint_matrix([smiles[int(i)] for i in idx_ok])
    K = tanimoto_gram(X, X)
    oof = np.empty(len(tv))
    for f in np.array_split(rng.permutation(len(tv)), 5):
        keep = np.setdiff1d(np.arange(len(tv)), f)
        gp = TanimotoGP().fit_gram(K[np.ix_(keep, keep)], tv[keep])
        oof[f] = gp.predict_gram(K[np.ix_(f, keep)], return_std=False)
    rho_gp = spearman0(oof, tv)
    rho_incr = spearman0(pv, tv - oof)

    print(f"\n    {'quantity':<44}{'Spearman':>10}")
    print("    " + "-" * 54)
    print(f"    {'Boltz-2 affinity vs measured pIC50':<44}{rho:>10.3f}")
    print(f"    {'Tanimoto GP (out of fold) vs measured':<44}{rho_gp:>10.3f}")
    print(f"    {'Boltz-2 vs what the GP gets wrong':<44}{rho_incr:>10.3f}")

    print("\n    against this project's own thresholds:")
    for label, thr in (("prior mean (FINDINGS 8)", PRIOR_MEAN_THRESHOLD),
                       ("auxiliary task (FINDINGS 32)", AUX_TASK_THRESHOLD)):
        print(f"      {label:<32} {thr:.2f}   "
              f"{'CLEARS' if rho >= thr else 'below'} ({rho:.3f})")

    print("\n    verdict:")
    if rho < PRIOR_MEAN_THRESHOLD:
        print("      Boltz-2 does not clear the lower threshold on this block. As a")
        print("      prior mean it would make low-budget BO worse; as an auxiliary")
        print("      task it would not separate from noise. It may still be usable")
        print("      as an *oracle* -- optimising it is a different claim from it")
        print("      predicting measured potency -- but it cannot substitute for a")
        print("      measured auxiliary.")
    elif rho < AUX_TASK_THRESHOLD:
        print("      Clears the prior-mean threshold but not the auxiliary-task one.")
        print("      Usable as a prior mean at low budget; not as a replacement for")
        print("      a co-measured panel.")
    else:
        print("      Clears both. A computed auxiliary is viable on this evidence,")
        print("      which would remove the co-measured-panel requirement that keeps")
        print("      the transfer result off a purchasable catalogue.")
    if abs(rho_incr) < 0.1:
        print("      Note: it explains almost nothing of what the fingerprint GP")
        print("      gets wrong, so even where it correlates it may be redundant.")

    report = {"target": args.target, "column": args.column, "family": args.family,
              "n_requested": int(args.n), "n_scored": int(ok.sum()),
              "sequence_length": len(sequence), "cache_key": oracle.cache_key,
              "deterministic": cfg.is_deterministic,
              "rho_boltz_vs_measured": rho, "rho_gp_oof_vs_measured": rho_gp,
              "rho_boltz_vs_gp_residual": rho_incr,
              "thresholds": {"prior_mean": PRIOR_MEAN_THRESHOLD,
                             "aux_task": AUX_TASK_THRESHOLD},
              "seconds": time.time() - t0,
              "pairs": [{"smiles": smiles[int(i)], "measured": float(truth[int(i)]),
                         "boltz2": float(pred[int(i)])} for i in idx_ok]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    oracle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Characterise the search space, and record what BO actually fits and suggests.

    python scripts/analyze_search_space.py

Writes three files the report reads:

``results/search_space.json``
    Geometry of the discrete space and of the kernel's feature space: how many
    fingerprint bits are ever set, how many carry information, the density, and
    the *effective* rank of the Tanimoto Gram. The gap between the nominal 2,048
    bits and the effective rank is the whole argument for using a Tanimoto
    kernel rather than a squared-exponential one.

``results/fp_collisions.json``
    How many distinct library members are *identical* to the surrogate, under
    both fingerprint settings. This is an irreducible noise floor -- a GP handed
    two identical inputs with different targets can only average them -- so it
    belongs in the report next to the results, not in a footnote.

``results/gp_trace_seed0.json``
    One real campaign, round by round: the GP hyperparameters as refitted from
    the data, the molecule chosen, what was predicted, and what the assay said.
    This is the evidence for "what does BO update, and what does it suggest".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from rdkit import DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from molbo.acquisition import acquisition_scores, rank_candidates  # noqa: E402
from molbo.chem import FINGERPRINT_VERSION, to_mol  # noqa: E402
from molbo.oracles import LookupOracle  # noqa: E402
from molbo.surrogate import LibraryKernel, TanimotoGP, fingerprint_matrix, tanimoto_gram  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
DATASET = "data/chembl/egfr_anilinoquinazoline.csv"


def load():
    orc = LookupOracle.from_dataset(DATASET, min_heavy_atoms=10, max_heavy_atoms=55)
    return orc, orc.library, np.asarray(orc.values)


def geometry(lib, vals) -> dict:
    X = fingerprint_matrix(lib)
    n, d = X.shape
    col = X.sum(axis=0)
    bits = X.sum(axis=1)
    K = tanimoto_gram(X, X)
    off = K[np.triu_indices(n, k=1)]
    ev = np.clip(np.linalg.eigvalsh(K)[::-1], 0, None)
    cum = np.cumsum(ev) / ev.sum()
    heavy = np.array([to_mol(s).GetNumHeavyAtoms() for s in lib])
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "n_molecules": int(n),
        "fp_nominal_dim": int(d),
        "fp_active_bits": int((col > 0).sum()),
        "fp_informative_bits": int(((col > 0) & (col < n)).sum()),
        "bits_per_mol_mean": float(bits.mean()),
        "bits_per_mol_min": int(bits.min()),
        "bits_per_mol_max": int(bits.max()),
        "density_pct": float(100 * bits.mean() / d),
        "tanimoto_mean": float(off.mean()),
        "tanimoto_median": float(np.median(off)),
        "tanimoto_p95": float(np.percentile(off, 95)),
        "tanimoto_max": float(off.max()),
        # Participation ratio: how many eigen-directions the kernel really uses.
        "eff_rank_participation": float(ev.sum() ** 2 / (ev**2).sum()),
        "eigs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "eigs_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "heavy_min": int(heavy.min()),
        "heavy_max": int(heavy.max()),
        "heavy_median": float(np.median(heavy)),
        "value_min": float(vals.min()),
        "value_max": float(vals.max()),
    }


def collisions(lib, vals) -> dict:
    """Count library members indistinguishable to the GP, with and without chirality."""
    out = {"note": "ECFP4 radius 2, 2048 bits", "library_size": len(lib)}
    for chiral in (False, True):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=chiral
        )
        X = np.zeros((len(lib), 2048), dtype=np.float32)
        for i, smi in enumerate(lib):
            mol = to_mol(smi)
            if mol is None:
                continue
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
            X[i] = arr
        K = tanimoto_gram(X, X)
        np.fill_diagonal(K, 0.0)
        i, j = np.where(K >= 0.9999)
        keep = i < j
        i, j = i[keep], j[keep]
        gaps = np.abs(vals[i] - vals[j])
        worst = []
        for a, b, g in sorted(zip(i, j, gaps), key=lambda t: -t[2])[:3]:
            worst.append({"gap": round(float(g), 2), "a": lib[a], "b": lib[b],
                          "value_a": float(vals[a]), "value_b": float(vals[b])})
        out[f"includeChirality_{str(chiral).lower()}"] = {
            "pairs": int(len(i)),
            "molecules": int(len(set(i.tolist()) | set(j.tolist()))),
            "max_gap": float(gaps.max()) if len(gaps) else 0.0,
            "mean_gap": float(gaps.mean()) if len(gaps) else 0.0,
            "pairs_gap_gt_half": int((gaps > 0.5).sum()),
            "worst_examples": worst,
        }
    return out


def gp_trace(lib, vals, seed=0, budget=50, n_init=10) -> list[dict]:
    """Replay one plain-BO campaign, recording the surrogate at every step."""
    X = fingerprint_matrix(lib)
    K = LibraryKernel(X)
    rng = np.random.default_rng(seed)
    evaluated = np.zeros(len(lib), bool)
    obs_idx, obs_val, rows = [], [], []

    for i in np.random.default_rng(seed).choice(len(lib), size=n_init, replace=False):
        evaluated[i] = True
        obs_idx.append(int(i))
        obs_val.append(float(vals[i]))

    rnd = 0
    while len(obs_idx) < budget:
        gp = TanimotoGP().fit_gram(K.train_block(obs_idx), np.asarray(obs_val))
        cand = np.flatnonzero(~evaluated)
        mu, sd = gp.predict_gram(K.cross_block(cand, obs_idx), k_diag=K.k_diag[cand])
        best = float(max(obs_val))
        sc = acquisition_scores("ei", mu, sd, best=best, rng=rng, xi=0.01)
        top = rank_candidates(sc, rng.random(sc.size) * 1e-12)[:4]
        pick = int(cand[top[0]])
        rows.append({
            "round": rnd, "n_obs": len(obs_idx),
            "eta": gp.eta_, "signal_var": round(gp.scale_, 4),
            "gp_mean": round(gp.mean_, 3), "y_scale": round(gp.y_scale_, 3),
            "lml": round(gp.lml_, 2), "best_so_far": round(best, 2),
            "suggested": lib[pick],
            "pred_mu": round(float(mu[top[0]]), 3),
            "pred_sd": round(float(sd[top[0]]), 3),
            "ei": round(float(sc[top[0]]), 4),
            "actual": round(float(vals[pick]), 2),
            "runner_up": [
                {"smiles": lib[int(cand[t])], "mu": round(float(mu[t]), 2),
                 "sd": round(float(sd[t]), 2), "ei": round(float(sc[t]), 4)}
                for t in top[1:]
            ],
        })
        evaluated[pick] = True
        obs_idx.append(pick)
        obs_val.append(float(vals[pick]))
        rnd += 1
    return rows


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    orc, lib, vals = load()
    print(f"library {len(lib):,} compounds, fingerprint {FINGERPRINT_VERSION}")

    geo = geometry(lib, vals)
    (RES / "search_space.json").write_text(json.dumps(geo, indent=2), encoding="utf-8")
    print(f"  effective rank {geo['eff_rank_participation']:.1f} of "
          f"{geo['fp_nominal_dim']:,} nominal dims; density {geo['density_pct']:.2f}%")

    col = collisions(lib, vals)
    (RES / "fp_collisions.json").write_text(json.dumps(col, indent=2), encoding="utf-8")
    a, b = col["includeChirality_false"], col["includeChirality_true"]
    print(f"  indistinguishable pairs: {a['pairs']} without chirality, "
          f"{b['pairs']} with (in use: {FINGERPRINT_VERSION})")

    tr = gp_trace(lib, vals)
    (RES / "gp_trace_seed0.json").write_text(json.dumps(tr, indent=2), encoding="utf-8")
    err = float(np.mean([abs(r["pred_mu"] - r["actual"]) for r in tr]))
    psd = float(np.mean([r["pred_sd"] for r in tr]))
    print(f"  gp trace: {len(tr)} rounds, eta values "
          f"{sorted({r['eta'] for r in tr})}, |error| {err:.2f} vs predicted sd {psd:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

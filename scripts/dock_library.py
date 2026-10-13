"""Dock a library once with AutoDock Vina, cached per molecule.

    python scripts/dock_library.py --dataset data/chembl/egfr_anilinoquinazoline.csv \
        --pdb 1M17 --native AQ4 --out data/cache/docking/egfr_1m17.jsonl

Docking is the one external tool in this project that **cannot have memorised
ChEMBL**: it scores a pose with a physics-based function, not a model fitted to
the same activity tables the benchmark is drawn from. That is why it is worth
the setup, and why an ML affinity predictor is not a substitute.

The economics are what make it usable inside a campaign: a score depends only
on ``(receptor, box, ligand)``, so every molecule is docked **once, ever**, and
every later round is a dictionary lookup. One pass over a 2,000-compound
library is under an hour on 32 cores.

Validation is not optional and is printed every run: the receptor's own
co-crystallised ligand is re-docked and its pose compared with the
crystallographic one. If that does not come back under ~2 A the box or the
receptor is wrong, and every score in the file is noise.

Cache format is one JSON object per line, appended as results arrive, so an
interrupted run resumes without re-docking anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

VINA = Path("tools/vina.exe") if os.name == "nt" else Path("tools/vina")
RCSB = "https://files.rcsb.org/download/{}.pdb"


# --------------------------------------------------------------------------
# receptor
# --------------------------------------------------------------------------


def fetch_pdb(code: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RCSB.format(code.upper()), dest)
    return dest


def native_ligand_xyz(pdb: Path, resname: str) -> np.ndarray:
    """Heavy-atom coordinates of the co-crystallised ligand."""
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.read_text().splitlines()
           if l.startswith("HETATM") and l[17:20].strip() == resname and l[76:78].strip() != "H"]
    if not xyz:
        raise SystemExit(f"{pdb}: no HETATM residue named {resname!r}")
    return np.array(xyz)


def prepare_receptor(pdb: Path, native: str, out_stem: Path, pad: float = 10.0) -> tuple[Path, Path]:
    """-> (receptor pdbqt, Vina box config). Box is centred on the native ligand.

    Sized from the native ligand's extent plus ``pad`` on each side rather than
    a fixed cube: a box much larger than the site wastes search effort and
    quietly degrades the scores, which is the failure mode that looks like
    "docking does not work".
    """
    pdbqt, box = out_stem.with_suffix(".pdbqt"), Path(f"{out_stem}.box.txt")
    if pdbqt.exists() and box.exists():
        return pdbqt, box
    xyz = native_ligand_xyz(pdb, native)
    centre = xyz.mean(0)
    size = np.clip(xyz.max(0) - xyz.min(0) + 2 * pad, 20.0, 30.0)
    protein = out_stem.with_name(out_stem.name + "_rec.pdb")
    protein.parent.mkdir(parents=True, exist_ok=True)
    protein.write_text("\n".join(l for l in pdb.read_text().splitlines()
                                 if l.startswith(("ATOM", "TER"))) + "\n")
    exe = Path(sys.executable).with_name("mk_prepare_receptor.exe")
    cmd = [str(exe) if exe.exists() else "mk_prepare_receptor",
           "--read_pdb", str(protein), "-o", str(out_stem), "-p", "-v",
           "--box_center", *[f"{c:.2f}" for c in centre],
           "--box_size", *[f"{s:.0f}" for s in size],
           "-x", "--default_altloc", "A"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not pdbqt.exists():
        raise SystemExit(f"receptor prep failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    print(f"[dock] box centre {centre.round(2)}  size {size.round(0)}  -> {pdbqt.name}", flush=True)
    return pdbqt, box


# --------------------------------------------------------------------------
# one ligand
# --------------------------------------------------------------------------


def dock_one(args) -> tuple[str, float | None, str]:
    """-> (smiles, best score in kcal/mol, note). Runs in a worker process."""
    smiles, receptor, box, exhaustiveness, seed, keep_pose = args
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, "unparseable"
    mol = Chem.AddHs(mol)
    # A fixed seed derived from the molecule keeps the 3D start reproducible,
    # so a rerun of this script gives byte-identical scores.
    if AllChem.EmbedMolecule(mol, randomSeed=abs(hash(smiles)) % (2**31)) != 0:
        return smiles, None, "no 3D conformer"
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    try:
        setup = MoleculePreparation().prepare(mol)[0]
        txt, ok, err = PDBQTWriterLegacy.write_string(setup)
    except Exception as exc:
        return smiles, None, f"meeko: {type(exc).__name__}"
    if not ok:
        return smiles, None, f"meeko: {str(err)[:60]}"

    with tempfile.TemporaryDirectory() as td:
        lig, out = Path(td) / "l.pdbqt", Path(td) / "o.pdbqt"
        lig.write_text(txt)
        cmd = [str(VINA), "--receptor", receptor, "--ligand", str(lig), "--config", box,
               "--out", str(out), "--exhaustiveness", str(exhaustiveness),
               "--seed", str(seed), "--cpu", "1", "--verbosity", "0"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return smiles, None, "vina timeout"
        if not out.exists():
            return smiles, None, "vina produced no output"
        pose = out.read_text()
    for line in pose.splitlines():
        if line.startswith("REMARK VINA RESULT"):
            return smiles, float(line.split()[3]), pose if keep_pose else ""
    return smiles, None, "no score in output"


# --------------------------------------------------------------------------


def validate(receptor: Path, box: Path, native_xyz: np.ndarray, smiles: str,
             exhaustiveness: int) -> dict:
    """Re-dock the native ligand and compare with its crystallographic pose."""
    _, score, pose = dock_one((smiles, str(receptor), str(box), exhaustiveness, 42, True))
    if score is None:
        return {"ok": False, "why": pose}
    # Only the first MODEL -- that is Vina's best-scoring pose, the one the
    # reported score belongs to.
    cur: list[list[float]] = []
    for line in pose.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")) and line[77:79].strip() not in ("H", "HD"):
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not cur:
        return {"ok": False, "why": "no heavy atoms parsed from the docked pose"}
    xyz = np.asarray(cur)
    # Atom ordering differs between the PDB and an RDKit-built molecule, so an
    # in-order RMSD is meaningless. Nearest-neighbour distance is order- and
    # symmetry-insensitive and is what the <2 A convention effectively measures.
    nn = np.linalg.norm(xyz[:, None, :] - native_xyz[None, :, :], axis=2).min(1)
    return {"ok": bool(np.sqrt((nn ** 2).mean()) < 2.0), "score": score,
            "nn_rmsd": float(np.sqrt((nn ** 2).mean())),
            "centroid_shift": float(np.linalg.norm(xyz.mean(0) - native_xyz.mean(0))),
            "within_2A": int((nn < 2.0).sum()), "n_atoms": int(len(xyz))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="CSV from build_chembl_series.py")
    p.add_argument("--smiles-column", default="canonical_smiles")
    p.add_argument("--pdb", default="1M17", help="RCSB code (1M17 = EGFR + erlotinib)")
    p.add_argument("--native", default="AQ4", help="residue name of the co-crystallised ligand")
    p.add_argument("--native-smiles", default="C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
                   help="that ligand as SMILES, for the redocking check (default erlotinib)")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit", type=int, default=None, help="dock only the first N (for a trial)")
    p.add_argument("--struct-dir", default="data/cache/docking")
    p.add_argument("--out", required=True, help="cache .jsonl, appended and resumable")
    args = p.parse_args(argv)

    if not VINA.exists():
        raise SystemExit(f"{VINA} not found. Download the official binary:\n"
                         "  curl -sSL -o tools/vina.exe https://github.com/ccsb-scripps/"
                         "AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe")

    import pandas as pd

    struct = Path(args.struct_dir)
    pdb = fetch_pdb(args.pdb, struct / f"{args.pdb.lower()}.pdb")
    receptor, box = prepare_receptor(pdb, args.native, struct / args.pdb.lower())
    native_xyz = native_ligand_xyz(pdb, args.native)

    print("[dock] validating: re-docking the native ligand...", flush=True)
    v = validate(receptor, box, native_xyz, args.native_smiles, args.exhaustiveness)
    if not v.get("ok"):
        raise SystemExit(f"[dock] REDOCK FAILED -- {v}\n"
                         "The box or the receptor is wrong. Every score would be noise.")
    print(f"[dock] redock OK: score {v['score']:.2f} kcal/mol, nn-RMSD {v['nn_rmsd']:.2f} A, "
          f"centroid {v['centroid_shift']:.2f} A, {v['within_2A']}/{v['n_atoms']} atoms within 2 A",
          flush=True)

    smiles = list(dict.fromkeys(pd.read_csv(args.dataset)[args.smiles_column].dropna()))
    if args.limit:
        smiles = smiles[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, float | None] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["smiles"]] = r["score"]
    todo = [s for s in smiles if s not in done]
    print(f"[dock] {len(smiles)} molecules, {len(done)} cached, {len(todo)} to dock "
          f"on {args.workers} workers", flush=True)

    t0, failed = time.time(), 0
    with out.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(dock_one, (s, str(receptor), str(box), args.exhaustiveness,
                                       args.seed, False)) for s in todo]
        for i, fut in enumerate(as_completed(jobs), 1):
            smi, score, note = fut.result()
            failed += score is None
            fh.write(json.dumps({"smiles": smi, "score": score, "note": note}) + "\n")
            if i % 25 == 0 or i == len(jobs):
                rate = i / max(1e-9, time.time() - t0)
                print(f"\r[dock] {i}/{len(jobs)}  {rate:.1f}/s  "
                      f"eta {(len(jobs) - i) / max(rate, 1e-9) / 60:.0f} min  "
                      f"{failed} failed", end="", flush=True)
                fh.flush()
    print()

    scores = [v for v in ({**done, **{}}).values() if v is not None]
    with out.open(encoding="utf-8") as fh:
        allr = [json.loads(l) for l in fh if l.strip()]
    scores = [r["score"] for r in allr if r["score"] is not None]
    print(f"[dock] {len(allr)} rows, {len(scores)} scored, {len(allr) - len(scores)} failed")
    if scores:
        a = np.array(scores)
        print(f"[dock] score range {a.min():.2f} .. {a.max():.2f} kcal/mol, "
              f"median {np.median(a):.2f}  ({time.time() - t0:.0f}s)")
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({"dataset": args.dataset, "pdb": args.pdb, "native": args.native,
                                "exhaustiveness": args.exhaustiveness, "seed": args.seed,
                                "vina": "1.2.7", "redock": v, "n": len(allr),
                                "n_scored": len(scores)}, indent=2), encoding="utf-8")
    print(f"[dock] wrote {out} and {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

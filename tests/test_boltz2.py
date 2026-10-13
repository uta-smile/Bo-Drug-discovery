"""Boltz-2 plumbing: determinism guarantees, input generation, cache identity.

Boltz itself is not installed in CI and needs a GPU, so nothing here runs a
prediction. What is tested is everything that decides *whether the numbers are
reproducible* -- the per-ligand seed, the gamma override, and the cache key --
because those are the parts that would silently corrupt a results table rather
than fail loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbo.oracles.boltz2 import (
    BOLTZ2_DEFAULT_GAMMA_0,
    Boltz2Config,
    Boltz2Oracle,
    ligand_seed,
)

SEQ = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNE"


@pytest.fixture
def oracle(tmp_path):
    return Boltz2Oracle(
        Boltz2Config(protein_sequence=SEQ, msa_path="prot.a3m"),
        work_dir=tmp_path,
        keep_outputs=True,
    )


# ---- per-ligand seed ------------------------------------------------------


def test_seed_is_stable_across_calls():
    assert ligand_seed("CCO") == ligand_seed("CCO")


def test_seed_is_stable_across_processes():
    """``hash()`` is randomised per process; a run-to-run-varying seed would
    reintroduce exactly the nondeterminism this exists to remove."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src');"
            "from molbo.oracles.boltz2 import ligand_seed; print(ligand_seed('CCO'))",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": "random", "PATH": ""},
    )
    assert int(out.stdout.strip()) == ligand_seed("CCO")


def test_different_ligands_get_different_seeds():
    assert ligand_seed("CCO") != ligand_seed("CCC")


def test_seed_is_in_range():
    for smi in ("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"):
        assert 0 <= ligand_seed(smi) < 2**31 - 1


def test_salt_redraws_every_seed():
    """The explicit way to run a noisy-oracle arm, rather than by accident."""
    assert ligand_seed("CCO", salt="arm2") != ligand_seed("CCO")


def test_the_command_carries_the_ligands_own_seed(oracle, tmp_path):
    cmd = oracle._command(tmp_path / "x.yaml", tmp_path, ligand_seed("CCO"))
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == str(ligand_seed("CCO"))


# ---- gamma ----------------------------------------------------------------


def test_gamma_defaults_to_zero_not_to_boltz_defaults():
    cfg = Boltz2Config(protein_sequence=SEQ)
    assert cfg.gamma_0 == 0.0 and cfg.gamma_min == 0.0
    assert cfg.gamma_0 != BOLTZ2_DEFAULT_GAMMA_0
    assert cfg.is_deterministic


def test_a_nonzero_gamma_is_reported_as_nondeterministic():
    assert not Boltz2Config(protein_sequence=SEQ, gamma_0=0.8).is_deterministic


def test_the_runner_patches_both_diffusion_param_classes(oracle):
    """gamma is a dataclass field, not a CLI flag, so it must be patched."""
    runner = oracle._runner.read_text(encoding="utf-8")
    assert "Boltz2DiffusionParams" in runner
    assert "BoltzDiffusionParams" in runner
    assert "gamma_0 = 0.0" in runner
    assert "gamma_min = 0.0" in runner


def test_the_runner_is_valid_python(oracle):
    compile(oracle._runner.read_text(encoding="utf-8"), "runner", "exec")


def test_the_runner_is_invoked_instead_of_the_boltz_entrypoint(oracle, tmp_path):
    cmd = oracle._command(tmp_path / "x.yaml", tmp_path, 1)
    assert cmd[1].endswith("_boltz_runner.py")
    assert cmd[2] == "predict"


# ---- cache identity -------------------------------------------------------


def test_gamma_is_part_of_the_cache_key(tmp_path):
    a = Boltz2Oracle(Boltz2Config(protein_sequence=SEQ), work_dir=tmp_path / "a")
    b = Boltz2Oracle(
        Boltz2Config(protein_sequence=SEQ, gamma_0=0.8), work_dir=tmp_path / "b"
    )
    assert a.cache_key != b.cache_key


@pytest.mark.parametrize(
    "field,value",
    [
        ("protein_sequence", SEQ + "AAA"),
        ("diffusion_samples_affinity", 3),
        ("sampling_steps", 50),
        ("step_scale", 1.2),
        ("affinity_mw_correction", True),
        ("use_potentials", True),
        ("seed_salt", "arm2"),
    ],
)
def test_anything_that_changes_the_number_changes_the_cache_key(tmp_path, field, value):
    base = Boltz2Config(protein_sequence=SEQ)
    other = Boltz2Config(**{**base.__dict__, field: value})
    assert base.key() != other.key()


def test_the_value_field_is_part_of_the_cache_key(tmp_path):
    cfg = Boltz2Config(protein_sequence=SEQ)
    a = Boltz2Oracle(cfg, work_dir=tmp_path / "a")
    b = Boltz2Oracle(cfg, work_dir=tmp_path / "b", value="affinity_probability_binary")
    assert a.cache_key != b.cache_key


def test_an_unknown_value_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="affinity_pred_value"):
        Boltz2Oracle(Boltz2Config(protein_sequence=SEQ), work_dir=tmp_path, value="vibes")


def test_describe_records_the_determinism_settings(oracle):
    d = oracle.describe()
    assert d["deterministic"] is True
    assert d["gamma_0"] == 0.0
    assert d["negated"] is True
    assert "sha256" in d["seed_policy"]


# ---- YAML input -----------------------------------------------------------


def test_yaml_declares_the_ligand_as_the_affinity_binder(oracle, tmp_path):
    out = tmp_path / "j.yaml"
    oracle._write_yaml(out, "CCO")
    text = out.read_text(encoding="utf-8")
    assert "affinity:" in text
    assert "binder: B" in text
    assert "smiles: 'CCO'" in text
    assert f"sequence: {SEQ}" in text


def test_smiles_is_quoted_so_yaml_cannot_mangle_it(oracle, tmp_path):
    """SMILES routinely contains ':', '#' and '[' -- all YAML syntax unquoted."""
    smi = "C#CC(=O)N1CCC[C@@H]1c1nc(-c2ccc(Oc3ccccc3)cc2)c[nH]1"
    out = tmp_path / "j.yaml"
    oracle._write_yaml(out, smi)
    assert f"smiles: '{smi}'" in out.read_text(encoding="utf-8")

    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert parsed["sequences"][1]["ligand"]["smiles"] == smi


def test_msa_path_is_emitted_when_given(oracle, tmp_path):
    out = tmp_path / "j.yaml"
    oracle._write_yaml(out, "CCO")
    assert "msa: prot.a3m" in out.read_text(encoding="utf-8")


def test_msa_server_flag_is_opt_in(tmp_path):
    plain = Boltz2Oracle(Boltz2Config(protein_sequence=SEQ), work_dir=tmp_path / "a")
    server = Boltz2Oracle(
        Boltz2Config(protein_sequence=SEQ, use_msa_server=True), work_dir=tmp_path / "b"
    )
    assert "--use_msa_server" not in plain._command(Path("x"), Path("o"), 1)
    assert "--use_msa_server" in server._command(Path("x"), Path("o"), 1)


# ---- failure handling -----------------------------------------------------


def test_a_missing_result_is_a_none_not_an_exception(oracle):
    """One unfoldable ligand must not kill a 200-call campaign."""
    assert oracle._run_one("CCO") is None


def test_a_failure_is_recorded_for_inspection(oracle):
    oracle._run_one("CCO")
    failures = list(Path(oracle.work_dir).rglob("FAILED.txt"))
    assert failures
    assert "CCO" in failures[0].read_text(encoding="utf-8")


# ---- value convention -----------------------------------------------------


def test_affinity_pred_value_is_negated(oracle):
    """Boltz reports log10(IC50) in uM -- lower is more potent -- and every
    tier in this project maximises."""
    stem = __import__("hashlib").sha256(b"CCO").hexdigest()[:16]
    d = Path(oracle.work_dir) / stem / f"boltz_results_{stem}" / "predictions" / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / f"affinity_{stem}.json").write_text(
        json.dumps({"affinity_pred_value": 1.5, "affinity_probability_binary": 0.9})
    )
    assert oracle._run_one("CCO") == -1.5


def test_binary_probability_is_not_negated(tmp_path):
    orc = Boltz2Oracle(
        Boltz2Config(protein_sequence=SEQ),
        work_dir=tmp_path,
        keep_outputs=True,
        value="affinity_probability_binary",
    )
    stem = __import__("hashlib").sha256(b"CCO").hexdigest()[:16]
    d = Path(tmp_path) / stem / f"boltz_results_{stem}" / "predictions" / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / f"affinity_{stem}.json").write_text(
        json.dumps({"affinity_pred_value": 1.5, "affinity_probability_binary": 0.9})
    )
    assert orc._run_one("CCO") == 0.9

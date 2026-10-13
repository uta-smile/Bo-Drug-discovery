"""What `RunDigest` has to guarantee for a paired contrast to mean anything.

The uniform-history experiments are read as differences between runs: measured
auxiliary activities against a Gaussian null (`--aux-quality 0`) and against a
permutation null (`--aux-shuffle`). That subtraction is only a paired
comparison if the runs assayed the same historical compounds from the same
initial designs. Until the digest existed the evidence for that was indirect --
equal history sizes, and an equal `reach` -- and neither implies equal rows:
two different draws of 150 compounds from 335 land at the same reach routinely.

So the digest hashes the rows themselves, and these tests pin the three
properties the audit (`scripts/history_pairing_audit.py`) reads off it:

* it is a hash of *content*, so equal draws collide and unequal ones do not,
  including when only the order within a cell differs;
* the auxiliary nulls do not touch the history draw or the initial design,
  which is the invariant that makes the pairing legitimate;
* the auxiliary values themselves *do* change under each null, which is the
  failure the equality checks alone would pass in silence.

The last two run the real driver on a 90-compound subsample, because the thing
under test is where the digest is fed from inside the campaign loop, not the
hashing. A unit test over `RunDigest` cannot catch a `digest.cell()` call that
drifts to the wrong side of the shuffle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense_block_campaign as dbc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# A cheap but real campaign: one target pair's worth of cells on a 90-compound
# subsample of CYP. About a second per run.
MICRO = ["--family", "cyp", "--subsample", "90", "--history", "10",
         "--budget", "30", "--batch", "10", "--n-init", "10", "--k", "5",
         "--repeats", "1", "--audit", "--arms", "stl", "mt"]


def digest_of(cells, *, values=None):
    """A digest over ``cells``: a list of (target, ti, r, [index arrays])."""
    d = dbc.RunDigest()
    for target, ti, r, idx in cells:
        vals = values(idx) if values else [np.zeros(len(a)) for a in idx]
        d.cell(target, ti, r, aux_idx=idx, aux_val=vals)
    return d.to_json()


def test_equal_draws_collide_and_unequal_draws_do_not():
    a = [("JAK1", 0, 0, [np.array([3, 1, 2]), np.array([7, 8])])]
    b = [("JAK1", 0, 0, [np.array([3, 1, 2]), np.array([7, 8])])]
    c = [("JAK1", 0, 0, [np.array([3, 1, 2]), np.array([7, 9])])]
    assert digest_of(a)["history"] == digest_of(b)["history"]
    assert digest_of(a)["history"] != digest_of(c)["history"]


def test_draw_order_is_part_of_the_mask():
    """The same set drawn in another order is another draw.

    The history is sliced positionally into one block per auxiliary target, so
    a permutation of it assigns different compounds to different relatives even
    though the set is identical.
    """
    a = [("JAK1", 0, 0, [np.array([1, 2, 3])])]
    b = [("JAK1", 0, 0, [np.array([3, 2, 1])])]
    assert digest_of(a)["history"] != digest_of(b)["history"]


def test_cell_identity_is_hashed_not_just_the_rows():
    a = [("JAK1", 0, 0, [np.array([1, 2])]), ("JAK2", 1, 0, [np.array([3, 4])])]
    swapped = [("JAK1", 0, 0, [np.array([3, 4])]),
               ("JAK2", 1, 0, [np.array([1, 2])])]
    assert digest_of(a)["history"] != digest_of(swapped)["history"]


def test_index_width_does_not_change_the_digest():
    """A digest is a claim about rows, not about how numpy happened to type
    them: `np.setdiff1d` and `rng.choice` do not agree on width across
    platforms."""
    wide = [("JAK1", 0, 0, [np.array([1, 2, 3], dtype=np.int64)])]
    narrow = [("JAK1", 0, 0, [np.array([1, 2, 3], dtype=np.int32)])]
    assert digest_of(wide)["history"] == digest_of(narrow)["history"]


def test_values_stream_is_independent_of_the_history_stream():
    idx = [np.array([1, 2, 3])]
    cells = [("JAK1", 0, 0, idx)]
    lo = digest_of(cells, values=lambda i: [np.zeros(len(i[0]))])
    hi = digest_of(cells, values=lambda i: [np.ones(len(i[0]))])
    assert lo["history"] == hi["history"]
    assert lo["aux_value"] != hi["aux_value"]


def test_per_target_digests_localise_a_difference():
    a = [("JAK1", 0, 0, [np.array([1, 2])]), ("JAK2", 1, 0, [np.array([3, 4])])]
    b = [("JAK1", 0, 0, [np.array([1, 2])]), ("JAK2", 1, 0, [np.array([3, 5])])]
    da, db = digest_of(a)["by_target"], digest_of(b)["by_target"]
    assert da["history"]["JAK1"] == db["history"]["JAK1"]
    assert da["history"]["JAK2"] != db["history"]["JAK2"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The driver, under each auxiliary condition and one changed seed."""
    out = tmp_path_factory.mktemp("digest")
    cases = {
        "real": [],
        "q0": ["--aux-quality", "0"],
        "shuf": ["--aux-shuffle"],
        "seed1": ["--aux-quality", "0", "--seed", "1"],
    }
    done = {}
    for name, extra in cases.items():
        path = out / f"{name}.json"
        assert dbc.main([*MICRO, *extra, "--out", str(path)]) in (0, None)
        done[name] = json.loads(path.read_text(encoding="utf-8"))
    return done


def test_nulls_leave_the_history_and_the_initial_design_alone(runs):
    base = runs["real"]["digest"]
    for name in ("q0", "shuf"):
        other = runs[name]["digest"]
        assert other["history"] == base["history"], name
        assert other["init_design"] == base["init_design"], name
        assert other["cells"] == base["cells"], name


def test_each_null_actually_changes_the_auxiliary_values(runs):
    seen = {name: runs[name]["digest"]["aux_value"]
            for name in ("real", "q0", "shuf")}
    assert len(set(seen.values())) == 3, seen


def test_a_different_seed_is_a_different_history(runs):
    """The complement of the test above: the digest is not simply constant."""
    assert runs["seed1"]["digest"]["history"] != runs["q0"]["digest"]["history"]
    assert runs["seed1"]["digest"]["init_design"] != runs["q0"]["digest"]["init_design"]


def test_the_digest_travels_with_the_artefact(runs):
    """A stored run has to carry its own digest, or a later comparison has
    nothing to compare -- which is the gap this closes."""
    d = runs["real"]["digest"]
    assert set(dbc.RunDigest.STREAMS) <= set(d)
    assert d["cells"] == runs["real"]["audit"]["cells"]
    assert all(len(d[s]) == 64 for s in dbc.RunDigest.STREAMS)

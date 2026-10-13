"""History-aware seeding: the rules, and the invariant they must not break.

The initial design is the one hook a language model has already been given in
this project, and the one it lost on -- `bo_llm_warmstart` beat plain BO and
then lost to a free MaxMin rule picking from the same sample. But that arm's
prompt says *"Nothing has been measured yet"*: it was never shown the auxiliary
history. These rules are that missing condition, written as free deterministic
baselines so that whatever an agent is asked later has a control standing in
front of it.

Two things are pinned below, and the second matters more than the first.

1. The rules do what they say -- `aux_top` is the history's own winners,
   `aux_scaffold_nn` is a *superset* of it, and neither reads a primary value.
2. `run_campaign(init_idx=None)` still draws exactly as it did. The parameter
   exists so `init_headroom.py` can score several designs on one rng stream;
   if its default path ever diverges, every stored campaign result was produced
   by different code than the one being tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dense_block_campaign import (  # noqa: E402
    history_seed_pool,
    run_campaign,
)
from init_headroom import maxmin_design  # noqa: E402


@pytest.fixture
def toy():
    """Twelve compounds, two auxiliaries, a similarity matrix with structure."""
    rng = np.random.default_rng(0)
    n = 12
    X = rng.random((n, 8)) < 0.5
    K = np.array([[float(np.sum(a & b) / max(np.sum(a | b), 1)) for b in X]
                  for a in X])
    np.fill_diagonal(K, 1.0)
    aux_idx = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]
    aux_val = [np.array([9.0, 1.0, 1.0, 1.0]), np.array([1.0, 8.0, 1.0, 1.0])]
    scaf = ["A", "B", "C", "D", "A", "B", "C", "D", "E", "E", "F", "F"]
    return K, scaf, aux_idx, aux_val


# ---- the rules -----------------------------------------------------------

def test_aux_top_takes_the_historys_own_winners(toy):
    K, scaf, aux_idx, aux_val = toy
    pool, info = history_seed_pool(K, scaf, aux_idx, aux_val, mode="aux_top",
                                   n_top=1, n_nn=0)
    # Compound 0 is the first auxiliary's best, compound 5 the second's.
    assert sorted(pool.tolist()) == [0, 5]
    assert info["n_winners"] == 2


def test_scores_are_z_scored_per_auxiliary(toy):
    """A relative whose values run higher must not supply the whole pool.

    Same ranks, one auxiliary shifted up by 100: the winners cannot move.
    """
    K, scaf, aux_idx, aux_val = toy
    shifted = [aux_val[0] + 100.0, aux_val[1]]
    a = history_seed_pool(K, scaf, aux_idx, aux_val, mode="aux_top",
                          n_top=2, n_nn=0)[0]
    b = history_seed_pool(K, scaf, aux_idx, shifted, mode="aux_top",
                          n_top=2, n_nn=0)[0]
    assert a.tolist() == b.tolist()


def test_scaffold_nn_is_a_superset_of_aux_top(toy):
    """The pair answers whether widening the seed region helps.

    That reading only holds if one contains the other; if the two pools could
    diverge, a difference between them would not be about width.
    """
    K, scaf, aux_idx, aux_val = toy
    tight = history_seed_pool(K, scaf, aux_idx, aux_val, mode="aux_top",
                              n_top=2, n_nn=0)[0]
    wide = history_seed_pool(K, scaf, aux_idx, aux_val,
                             mode="aux_scaffold_nn", n_top=2, n_nn=2)[0]
    assert set(tight.tolist()) <= set(wide.tolist())


def test_scaffold_nn_pulls_in_scaffold_mates(toy):
    """Compound 0's scaffold "A" is shared with compound 4, which has no
    history of its own -- reaching it is the whole point of the rule."""
    K, scaf, aux_idx, aux_val = toy
    wide = history_seed_pool(K, scaf, aux_idx, aux_val,
                             mode="aux_scaffold_nn", n_top=1, n_nn=0)[0]
    assert 4 in wide.tolist()


def test_unknown_mode_is_an_error(toy):
    K, scaf, aux_idx, aux_val = toy
    with pytest.raises(ValueError):
        history_seed_pool(K, scaf, aux_idx, aux_val, mode="nope", n_top=1,
                          n_nn=1)


def test_maxmin_picks_distinct_spread_out_compounds(toy):
    K, *_ = toy
    d = maxmin_design(K, 5, np.random.default_rng(1))
    assert len(set(d.tolist())) == 5
    # Every pick after the first is at least as far from the set as the
    # alternatives were when it was chosen; the cheap observable is that no
    # pair is more similar than the library's own median.
    sims = [K[a, b] for i, a in enumerate(d) for b in d[i + 1:]]
    off = K[~np.eye(len(K), dtype=bool)]
    assert np.median(sims) <= np.median(off) + 1e-9


# ---- the invariant -------------------------------------------------------

def _campaign(**over):
    rng_seed = over.pop("rng_seed", 3)
    n = 40
    g = np.random.default_rng(0)
    X = g.random((n, 16)) < 0.5
    K = np.array([[float(np.sum(a & b) / max(np.sum(a | b), 1)) for b in X]
                  for a in X])
    np.fill_diagonal(K, 1.0)
    y = g.random(n)
    kw = dict(arm="stl", budget=16, batch=4, n_init=6, k=5, acquisition="ei",
              rng=np.random.default_rng(rng_seed))
    kw.update(over)
    return run_campaign(K, y, [], [], **kw)


def test_default_path_is_unchanged_by_the_new_parameter():
    """`init_idx=None` must draw exactly as before.

    Every stored campaign number was produced on this path. If adding the
    parameter perturbed it, the comparison between old and new results would be
    silently between two different programs.
    """
    a_seq, a_idx = _campaign()
    b_seq, b_idx = _campaign(init_idx=None)
    assert np.array_equal(a_seq, b_seq)
    assert np.array_equal(a_idx, b_idx)


def test_init_idx_is_used_verbatim_and_truncated_to_n_init():
    _, idx = _campaign(init_idx=np.array([9, 8, 7, 6, 5, 4, 3, 2]))
    assert idx[:6].tolist() == [9, 8, 7, 6, 5, 4]


def test_two_designs_differ_only_in_where_they_started():
    """The point of the parameter: same rng stream, different start.

    Without it, scoring two initial designs means two rng streams, and the
    tie-breaking jitter downstream varies along with the thing under test.
    """
    a_seq, a_idx = _campaign(init_idx=np.arange(6))
    b_seq, b_idx = _campaign(init_idx=np.arange(6, 12))
    assert a_idx[:6].tolist() != b_idx[:6].tolist()
    assert len(a_seq) == len(b_seq) == 16

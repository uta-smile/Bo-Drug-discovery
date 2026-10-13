"""CLM selector plumbing. No backbone: a fake encoder stands in for Qwen."""

import hashlib

import numpy as np
import pytest

from molbo.clm import (
    CLMSelector,
    Heads,
    MaxMeanSelector,
    RecordingSelector,
    shortlist_metrics,
)

torch = pytest.importorskip("torch")

HIDDEN = 32
CFG = dict(hidden=HIDDEN, width=16, depth=3, proj=8, activation="gelu",
           layernorm=True, residual=False, encoder="fake", tau=0.25)


class FakeEncoder:
    def key(self):
        return "fake"

    def embed(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(HIDDEN).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_shortlist_metrics_reference_points():
    true = [np.array([1.0, 3.0, 2.0])]
    oracle = shortlist_metrics(true, true)
    assert oracle["regret"] == 0 and oracle["top1"] == 1
    assert oracle["gain"] == pytest.approx(1.0)
    worst = shortlist_metrics([-true[0]], true)
    assert worst["regret"] == pytest.approx(2.0) and worst["top1"] == 0


def test_recording_selector_is_plain_bo_and_labels_from_table():
    rec = RecordingSelector(values={"A": 5.0, "B": 7.0}, tag="t")
    pick = rec(shortlist=["B", "A"], scores=[0.2, 0.1], mu=[6, 5], sd=[1, 1],
               observations=[("C", 4.0)], round_index=3, seed=1)
    assert pick == 0
    r = rec.records[0]
    assert r["true"] == [7.0, 5.0] and r["round"] == 3 and r["observations"] == [["C", 4.0]]


def test_max_mean_selector():
    assert MaxMeanSelector()(mu=np.array([0.1, 0.9, 0.5])) == 1


def test_heads_roundtrip_and_selector(tmp_path):
    heads = Heads(CFG, device="cpu")
    path = tmp_path / "h.pt"
    heads.save(path, note="x")
    loaded = Heads.load(path, device="cpu")
    assert loaded.meta["note"] == "x"

    sel = CLMSelector(heads=loaded, encoder=FakeEncoder())
    shortlist = ["CCO", "c1ccccc1", "CCN", "CCC"]
    obs = [("CCCl", 6.0), ("CCBr", 5.0)]
    a = sel.score(shortlist, obs)
    b = CLMSelector(heads=heads.train(False), encoder=FakeEncoder()).score(shortlist, obs)
    np.testing.assert_allclose(a, b, rtol=1e-5)

    rank = sel(shortlist=shortlist, observations=obs)
    assert rank == int(np.argmax(a)) and sel.ranks == [rank]
    assert sum(sel.probs[0]) == pytest.approx(1.0, abs=1e-3)


def test_selector_is_order_invariant():
    """The property the chat selector lacked: a candidate's score does not
    depend on where it sits in the list.

    The candidates go through ``Heads.logits`` as one batch, so the property is
    exact in real arithmetic but only up to reduction order in float32: a
    batched matmul may sum its products in a different order when the BLAS
    thread count changes, and this assertion at ``rtol=1e-5`` was observed
    failing once under heavy parallel load and passing on every serial run.
    Threads are pinned for the comparison so the reduction order is fixed,
    which removes the flake without weakening what is being asserted -- the
    tolerance stays where it was.
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sel = CLMSelector(heads=Heads(CFG, device="cpu").train(False),
                          encoder=FakeEncoder())
        shortlist = ["CCO", "c1ccccc1", "CCN", "CCC", "CCOC"]
        obs = [("CCCl", 6.0)]
        s = sel.score(shortlist, obs)
        perm = [3, 0, 4, 1, 2]
        s_perm = sel.score([shortlist[i] for i in perm], obs)
        np.testing.assert_allclose(s_perm, s[perm], rtol=1e-5)
    finally:
        torch.set_num_threads(n_threads)


def test_from_checkpoint_rejects_mismatched_encoder(tmp_path):
    path = tmp_path / "h.pt"
    Heads(CFG, device="cpu").save(path)

    class Other(FakeEncoder):
        def key(self):
            return "other"

    with pytest.raises(ValueError, match="trained on"):
        CLMSelector.from_checkpoint(path, encoder=Other())

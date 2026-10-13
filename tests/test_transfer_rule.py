"""The out-of-sample rule must be able to lose.

Every threshold in this package so far was fitted to the blocks it is quoted
against. `transfer_rule.py` is the version that can fail: it fits on `n - 1`
blocks and spends the held-out one. A validation harness that cannot report
"the rule is not worth using" is not a validation harness, so these tests pin
the three outcomes it has to be able to produce --

1. a feature that genuinely separates the options -> the rule beats the best
   fixed policy,
2. a feature that is pure noise -> it does not,
3. and the held-out block is never in the fit.

The third is the one that would be silent if it broke.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import transfer_rule as tr  # noqa: E402


def make(tmp_path, rhos, *, transfer_of_rho, seed_of_rho=None, n_camp=40,
         noise=0.0, seed=0):
    """A synthetic sweep on disk, in the artefact shape the script reads."""
    res = tmp_path / "res"
    res.mkdir(exist_ok=True)
    seed_dir = tmp_path / "seedres"
    blocks = {"blocks": [{"rho": float(r)} for r in rhos]}
    (tmp_path / "blocks.json").write_text(json.dumps(blocks), encoding="utf-8")
    rng = np.random.default_rng(seed)
    for i, r in enumerate(rhos):
        label = f"k{i + 1:02d}"
        base = np.full(n_camp, 10.0)
        mt = base + transfer_of_rho(r) + rng.normal(0, noise, n_camp)
        (res / f"{label}.json").write_text(json.dumps({
            "n_pool": 400,
            "arms": {"stl": {"auc": base.tolist()}, "mt": {"auc": mt.tolist()}},
        }), encoding="utf-8")
        if seed_of_rho is not None:
            seed_dir.mkdir(exist_ok=True)
            sd = base + seed_of_rho(r) + rng.normal(0, noise, n_camp)
            (seed_dir / f"{label}.json").write_text(json.dumps({
                "arms": {"stl": {"auc": sd.tolist()}}}), encoding="utf-8")
    return res, tmp_path / "blocks.json", (seed_dir if seed_of_rho else None)


def test_a_real_signal_lets_the_rule_win(tmp_path):
    """Transfer pays above rho = 0.3 and costs below it. A rule reading rho
    should beat both fixed policies, because neither is right everywhere."""
    rhos = np.linspace(-0.1, 0.6, 14)
    res, blocks, _ = make(tmp_path, rhos, transfer_of_rho=lambda r: 4 * (r - 0.3))
    rows = tr.load(res, blocks, None)
    opts = tr.options_present(rows)
    fixed = {o: np.mean([r[o].mean() for r in rows]) for o in opts}
    earned = []
    for i, test in enumerate(rows):
        pick, _ = tr.fit_predict(rows[:i] + rows[i + 1:], test, opts, "rho")
        earned.append(test[pick].mean())
    assert np.mean(earned) > max(fixed.values()) + 1e-9


def test_a_useless_feature_does_not_let_it_win(tmp_path):
    """Transfer pays the same everywhere, so rho carries nothing. The rule must
    not manufacture an advantage -- at best it ties always-transfer."""
    rng = np.random.default_rng(1)
    rhos = rng.random(14)
    res, blocks, _ = make(tmp_path, rhos, transfer_of_rho=lambda r: 0.5,
                          noise=1.0, seed=2)
    rows = tr.load(res, blocks, None)
    opts = tr.options_present(rows)
    fixed = {o: np.mean([r[o].mean() for r in rows]) for o in opts}
    earned = []
    for i, test in enumerate(rows):
        pick, _ = tr.fit_predict(rows[:i] + rows[i + 1:], test, opts, "rho")
        earned.append(test[pick].mean())
    assert np.mean(earned) <= max(fixed.values()) + 1e-9


def test_the_held_out_block_is_not_in_the_fit(tmp_path):
    """The leak that would be invisible. One block is given an enormous gain;
    if it were in its own training set the fit would bend toward it, so the
    prediction made for it must not change when its own value is altered."""
    rhos = np.linspace(0.0, 0.5, 10)
    res, blocks, _ = make(tmp_path, rhos, transfer_of_rho=lambda r: 2 * r)
    rows = tr.load(res, blocks, None)
    opts = tr.options_present(rows)
    _, before = tr.fit_predict(rows[1:], rows[0], opts, "rho")
    rows[0]["transfer"] = rows[0]["transfer"] + 1000.0
    _, after = tr.fit_predict(rows[1:], rows[0], opts, "rho")
    assert before == after


def test_abstain_is_zero_by_construction(tmp_path):
    res, blocks, _ = make(tmp_path, [0.1, 0.2, 0.3, 0.4],
                          transfer_of_rho=lambda r: r)
    rows = tr.load(res, blocks, None)
    for r in rows:
        assert np.all(r["abstain"] == 0.0)


def test_the_third_option_is_picked_up_when_present(tmp_path):
    res, blocks, seed_dir = make(
        tmp_path, np.linspace(0, 0.5, 8),
        transfer_of_rho=lambda r: 4 * (r - 0.3),
        seed_of_rho=lambda r: 0.4)
    rows = tr.load(res, blocks, seed_dir)
    assert tr.options_present(rows) == ("abstain", "seed", "transfer")


def test_a_short_sweep_refuses_rather_than_reporting(tmp_path):
    res, blocks, _ = make(tmp_path, [0.1, 0.2], transfer_of_rho=lambda r: r)
    with pytest.raises(SystemExit, match="leave-one-out"):
        tr.main(["--results", str(res), "--blocks", str(blocks),
                 "--seed-results", str(tmp_path / "nope"),
                 "--out", str(tmp_path / "o.json")])


def test_zero_differences_are_counted_not_discarded_silently():
    """The rule picks the fixed policy on most blocks, so the paired
    difference is exactly zero there. scipy drops those pairs *and* loses its
    exact path, which is how this comparison was first quoted as p = 0.11 over
    eighteen blocks when the test saw three. The audit must surface both."""
    root = Path(__file__).resolve().parents[1]
    for name, best in (("uniform", "seed"), ("withheld", "transfer")):
        f = root / "results" / f"transfer_rule_{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        t = d["rule_vs_best_fixed"]
        assert t["best_fixed"] == best
        assert t["n_blocks"] == 18
        assert t["n_zero"] + t["n_discordant"] == 18
        assert t["n_discordant"] < t["n_blocks"], "zeros must be reported"
        # The exact test on the discordant blocks is the one to quote, and it
        # is weaker than scipy's default -- that is the whole point.
        assert t["signed_rank_p_exact"] >= t["scipy_default_p"]
        assert t["signed_rank_p_exact"] >= 2.0 ** (1 - t["n_discordant"])

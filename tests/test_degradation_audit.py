"""The two claims the manuscript made about the noise mixture were false.

`degrade_to_rho` standardises the noise but does not orthogonalise it against
the signal, so the realised ``a = Corr(z, e)`` is O(1/sqrt(n)) and not zero.
The paper asserted that the construction preserves the column's standard
deviation and sets the sample Pearson correlation to ``q`` exactly. Neither
holds. These tests pin the correct closed forms against the implementation, so
that the next person to describe the operator describes the one that runs.

They also pin the part that *is* exact and that the controls depend on: both
correction terms carry a factor ``q * sqrt(1 - q^2)``, so they vanish at q = 0
and q = 1, which is where every control in the paper sits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from molbo.metrics import degrade_to_rho  # noqa: E402
import degradation_audit as da  # noqa: E402


def column(n=400, seed=1):
    g = np.random.default_rng(seed)
    return g.normal(size=n) + 0.6 * g.standard_exponential(size=n)


@pytest.mark.parametrize("q", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_closed_forms_describe_the_implementation(q):
    """The advisor's two expressions, against the code, at machine precision.

    Reproducing the implementation's own noise draw is the point: `a` has to
    be the one the mixture actually used, not an independent sample.
    """
    y = column()
    sd, mean = y.std(), y.mean()
    z = (y - mean) / sd
    g = np.random.default_rng(7)
    e = g.normal(size=len(z))
    e = (e - e.mean()) / e.std()

    root = np.sqrt(max(0.0, 1.0 - q ** 2))
    yt = (q * z + root * e) * sd + mean

    a = float(np.corrcoef(z, e)[0, 1])
    var_ratio, corr = da.predicted(q, a)

    assert yt.var() / y.var() == pytest.approx(var_ratio, abs=1e-10)
    assert np.corrcoef(yt, y)[0, 1] == pytest.approx(corr, abs=1e-10)


def test_q_is_not_the_achieved_correlation():
    """The claim that was wrong. At q = 0.5 the achieved value misses q by
    enough to matter, draw to draw, and the variance is not preserved."""
    y = column()
    off, var_off = [], []
    for seed in range(200):
        yt = degrade_to_rho(y, 0.5, np.random.default_rng(seed))
        off.append(np.corrcoef(yt, y)[0, 1] - 0.5)
        var_off.append(yt.var() / y.var() - 1.0)
    off, var_off = np.asarray(off), np.asarray(var_off)
    # Unbiased, but far from exact on any single draw.
    assert abs(off.mean()) < 0.01
    assert off.std() > 0.01, "if this is ~0 the operator was orthogonalised"
    assert np.abs(var_off).max() > 1e-6, "variance is not preserved exactly"


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_the_endpoints_are_exact(q):
    """Where every control in the paper sits. The correction terms carry
    q*sqrt(1-q^2), so both vanish and the variance ratio is exactly one."""
    y = column()
    for seed in range(20):
        yt = degrade_to_rho(y, q, np.random.default_rng(seed))
        assert yt.var() / y.var() == pytest.approx(1.0, abs=1e-9)
        if q == 1.0:
            assert np.corrcoef(yt, y)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_q_zero_leaves_a_small_random_correlation_not_a_zero_one():
    """`q = 0` removes the designed signal but leaves Corr = a, which is
    O(1/sqrt(n)). This is why the campaign measures and stores the achieved
    value instead of assuming it."""
    y = column(n=335)
    got = [np.corrcoef(degrade_to_rho(y, 0.0, np.random.default_rng(s)), y)[0, 1]
           for s in range(200)]
    got = np.asarray(got)
    assert abs(got.mean()) < 0.02
    assert 0.03 < got.std() < 0.09      # ~1/sqrt(335) = 0.055
    assert np.abs(got).max() > 0.05     # single draws are not negligible


def test_audit_runs_and_refuses_if_the_forms_stop_matching(capsys):
    assert da.main(["--n", "120", "--draws", "20"]) == 0
    out = capsys.readouterr().out
    assert "mixing parameter" in out
    assert "machine precision" in out


@pytest.mark.parametrize("q", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_the_two_corrections_do_not_share_a_factor(q):
    """An earlier draft said both deviations carry q*qbar. Only the variance
    one does. The correlation deviation is qbar**3 * a to first order: it
    equals a at q = 0, and vanishes only at q = 1."""
    a = 0.03
    qb = np.sqrt(max(0.0, 1.0 - q ** 2))
    var_dev, corr_dev = da.predicted(q, a)
    var_dev -= 1.0
    corr_dev -= q

    assert var_dev == pytest.approx(2 * q * qb * a, abs=1e-12)
    assert corr_dev == pytest.approx(qb ** 3 * a, abs=5e-4)

    if q in (0.0, 1.0):
        assert var_dev == pytest.approx(0.0, abs=1e-12)
    if q == 0.0:
        assert corr_dev == pytest.approx(a, abs=1e-12),             "at q = 0 the achieved correlation is a, not zero"
    if q == 1.0:
        assert corr_dev == pytest.approx(0.0, abs=1e-12)

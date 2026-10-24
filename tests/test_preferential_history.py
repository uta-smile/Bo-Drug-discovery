"""Preferentially sampled auxiliary history: the design axis C4 does not span.

C4 contrasts a uniform history with one the primary target's top-k has been
removed from. Both assume the panel was assembled without reference to how
active the compounds turned out to be. Real panels are not assembled that way
-- a compound is assayed because somebody expected it to be interesting -- so
actives are over-represented by the design rather than by chance. That is
preferential sampling in the sense of Diggle, Menezes & Su (JRSS-C
59(2):191-232, 2010), where ignoring it biases ordinary kriging.

`--history-prefer-aux ALPHA` adds the missing direction. What has to hold:

1. alpha = 0 takes the original code path, bit for bit, or every stored run
   silently changes;
2. alpha > 0 actually enriches, measured by how many of the primary's true
   top-k land in the history;
3. the score reads auxiliary columns only, so the design stays feasible --
   this is what separates it from `--history-excludes-top`, which needs the
   answer key and is therefore a stress test rather than a design;
4. it composes with `exclude_top` rather than fighting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dense_block_campaign as dbc  # noqa: E402
from molbo.surrogate import tanimoto_gram  # noqa: E402


def _block(seed=0, n=240, d=64):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, d)) < 0.12).astype(np.float64)
    X[X.sum(1) == 0, 0] = 1.0
    K = tanimoto_gram(X, X)
    latent = rng.multivariate_normal(np.zeros(n), K + 1e-6 * np.eye(n))
    y = latent + 0.3 * rng.normal(size=n)
    # Auxiliary score correlated with the primary, as a related panel is.
    prefer = latent + 0.5 * rng.normal(size=n)
    return K, y, prefer


def _draw(K, y, prefer, alpha, seed=11, size=90, exclude_top=False, k=10):
    return dbc.draw_history(
        K, y, size, np.random.default_rng(seed), cluster_frac=0.0,
        exclude_top=exclude_top, k=k, prefer=prefer, prefer_alpha=alpha)


def test_alpha_zero_is_the_original_code_path():
    """The standing constraint: every stored run predates this parameter."""
    K, y, prefer = _block()
    base = dbc.draw_history(K, y, 90, np.random.default_rng(11),
                            cluster_frac=0.0, exclude_top=False, k=10)
    same = _draw(K, y, prefer, 0.0)
    none = _draw(K, y, None, 2.0)
    assert np.array_equal(base, same)
    assert np.array_equal(base, none)


def test_preference_enriches_the_history_for_actives():
    """The knob has to move the quantity it is named for."""
    K, y, prefer = _block()
    caps = {}
    for alpha in (0.0, 1.0, 3.0):
        got = [dbc.topk_capture(y, _draw(K, y, prefer, alpha, seed=s), 10)
               for s in range(24)]
        caps[alpha] = float(np.mean(got))
    assert caps[0.0] < caps[1.0] < caps[3.0], caps
    # And the uniform design already captures more than chance, because
    # related targets share actives -- the baseline this is measured against
    # is not zero.
    assert caps[0.0] > 10 * 90 / 240 * 0.5


def test_preference_reads_no_primary_value():
    """Feasibility, which is the whole point of preferring on the auxiliary
    score rather than on the primary top-k: changing every primary value must
    leave the drawn history identical."""
    K, y, prefer = _block()
    a = _draw(K, y, prefer, 2.0)
    b = _draw(K, -3.0 * y + 17.0, prefer, 2.0)
    assert np.array_equal(a, b)


def test_preference_composes_with_exclude_top():
    """`exclude_top` removes the primary's best from the pool; enrichment then
    works on what is left. The combination is a meaningful cell -- a panel
    assembled towards auxiliary actives that happens to miss the primary's --
    so it must run and must still honour the exclusion."""
    K, y, prefer = _block()
    hist = _draw(K, y, prefer, 3.0, exclude_top=True)
    assert dbc.topk_capture(y, hist, 10) == 0
    assert len(set(hist.tolist())) == len(hist) == 90


def test_draw_is_deterministic_under_preference():
    K, y, prefer = _block()
    assert np.array_equal(_draw(K, y, prefer, 2.0), _draw(K, y, prefer, 2.0))


def test_large_alpha_does_not_overflow():
    """The softmax is shifted by its max before exponentiating; without that
    an alpha a user might plausibly try returns NaN probabilities."""
    K, y, prefer = _block()
    hist = _draw(K, y, prefer, 80.0)
    assert len(set(hist.tolist())) == 90

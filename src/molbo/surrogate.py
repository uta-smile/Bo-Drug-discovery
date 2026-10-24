"""Exact GP regression with a Tanimoto kernel over binary fingerprints.

Why not BoTorch: at the budgets this project runs (200 observations, a library
of a few thousand), an exact GP is a 200x200 Cholesky.  Fitting it in numpy is
microseconds, has no torch dependency, and -- the part that matters -- is
bit-for-bit deterministic given a seed, so a difference between two arms is a
difference between two methods and not between two nondeterministic draws.

The kernel is Tanimoto (MinMax on binary vectors)::

    k(x, y) = s * <x, y> / (|x|^2 + |y|^2 - <x, y>)

which is the standard choice for fingerprint GPs: it is a valid PSD kernel on
binary vectors and it matches how medicinal chemists actually judge similarity.
There is no lengthscale to fit, which removes the usual failure mode of a
squared-exponential GP on 2048 sparse binary dimensions collapsing to either
"everything is identical" or "nothing is related".

Hyperparameters are the signal variance ``s`` and the noise-to-signal ratio
``eta``.  For fixed ``eta`` the maximiser of the exact log marginal likelihood
over ``s`` is closed-form, so fitting is a 1-D scan over a fixed ``eta`` grid
with an analytic inner solve -- no gradients, no optimiser seed, no restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import copy

import numpy as np
from scipy.linalg import solve_triangular

from .chem import morgan_fingerprint

# Noise-to-signal ratios scanned during :meth:`TanimotoGP.fit`. Spans "the assay
# is near-exact" to "most of the variance is noise"; pIC50 replicate spread puts
# a real congeneric series near the middle.
ETA_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)

_JITTER = 1e-8


def fingerprint_matrix(smiles_list, n_bits: int = 2048) -> np.ndarray:
    """``(n, n_bits)`` float32 ECFP4 matrix. Unparseable rows are all-zero.

    Float32 rather than uint8 because every downstream op is a BLAS dot
    product; the cast would otherwise happen on every kernel evaluation.
    """
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi)
        if fp is not None:
            out[i] = fp
    return out


def tanimoto_gram(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity between rows of ``A`` and ``B``.

    Both are assumed binary. An all-zero row (an unparseable molecule) has zero
    similarity to everything including itself, which is the honest answer -- it
    carries no information -- and keeps the Gram matrix PSD.
    """
    inner = A @ B.T
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    denom = a2 + b2 - inner
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, inner / denom, 0.0)
    return np.asarray(out, dtype=np.float64)


@dataclass
class TanimotoGP:
    """Exact GP regression on fingerprints, fitted by marginal likelihood.

    Attributes set by :meth:`fit`: ``mean_``, ``scale_`` (signal variance),
    ``eta_`` (noise-to-signal ratio), ``lml_`` (log marginal likelihood).

    Parameters
    ----------
    standardize:
        Fit on z-scored targets and undo the transform on predict. Keeps the
        single ``scale_`` hyperparameter well-conditioned across oracles whose
        values live on wildly different scales (pIC50 ~ 4-11, PMO ~ 0-1).
    eta_grid:
        Noise-to-signal ratios to scan. A single-element tuple pins the noise.
    """

    standardize: bool = True
    eta_grid: tuple[float, ...] = ETA_GRID
    jitter: float = _JITTER
    fixed_scale: float | None = None
    """Override the profiled signal variance ``scale_``.

    ``scale_`` is normally profiled out as ``z'(K+eta I)^-1 z / n``, which for a
    joint fit is a quantity *pooled over every task*. That makes the primary
    task's predictive uncertainty depend on the auxiliary rows even when the
    task covariance is diagonal and the auxiliary rows cannot touch the
    posterior mean. Pinning it is what separates "the auxiliary data changed the
    prediction" from "the auxiliary data changed the uncertainty scale"."""

    X_: np.ndarray | None = field(default=None, repr=False)
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _L: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TanimotoGP":
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] != np.asarray(y).ravel().shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {np.asarray(y).ravel().shape[0]}")
        self.X_ = X
        return self._fit_gram(tanimoto_gram(X, X), y)

    def fit_gram(self, K: np.ndarray, y: np.ndarray) -> "TanimotoGP":
        """Fit from a precomputed train-train Tanimoto Gram matrix.

        For a fixed library the whole Gram can be computed once and sliced per
        round, which turns the dominant cost of a campaign -- re-deriving the
        same similarities at every BO step -- into array indexing. See
        :class:`LibraryKernel`.
        """
        self.X_ = None
        return self._fit_gram(np.asarray(K, dtype=np.float64), y)

    def _fit_gram(self, K0: np.ndarray, y: np.ndarray) -> "TanimotoGP":
        y = np.asarray(y, dtype=np.float64).ravel()
        if K0.shape[0] != y.shape[0]:
            raise ValueError(f"K has {K0.shape[0]} rows but y has {y.shape[0]}")
        if y.size == 0:
            raise ValueError("cannot fit a GP with no observations")

        K0 = K0.copy()
        self.mean_ = float(y.mean())
        sd = float(y.std())
        # A constant y (every molecule scored the same) has no signal to model;
        # fall back to a unit scale so predict() returns the mean with full
        # prior variance rather than dividing by zero.
        self.y_scale_ = sd if (self.standardize and sd > 1e-12) else 1.0
        z = (y - self.mean_) / self.y_scale_

        K0[np.diag_indices_from(K0)] += self.jitter
        n = z.size

        best = None
        for eta in self.eta_grid:
            try:
                L = np.linalg.cholesky(K0 + eta * np.eye(n))
            except np.linalg.LinAlgError:  # pragma: no cover - jitter covers this
                continue
            # Triangular solve, not the general LU one: L is already a Cholesky
            # factor, so this is O(n^2) instead of O(n^3). At 200 observations
            # refitted every round that single distinction is most of a
            # campaign's runtime.
            v = solve_triangular(L, z, lower=True, check_finite=False)
            quad = float(v @ v)
            # Profile out the signal variance: the LML derivative in s vanishes
            # at s* = z' (K0 + eta I)^-1 z / n, so only the eta scan is numeric.
            profiled = quad / n
            scale = (self.fixed_scale if self.fixed_scale is not None
                     else max(profiled, 1e-12))
            logdet = 2.0 * float(np.log(np.diag(L)).sum())
            # The data-fit term is z' (K0 + eta I)^-1 z / s. It collapses to the
            # constant `n` only at s = s*; a pinned (or floored) scale has to
            # evaluate it, or the scan degenerates into minimising `logdet` and
            # returns the smallest eta in the grid whatever the data says.
            fit = n if scale == profiled else quad / scale
            lml = -0.5 * (n * np.log(scale) + logdet + fit + n * np.log(2 * np.pi))
            if best is None or lml > best[0]:
                best = (lml, eta, scale, L, v)

        if best is None:  # pragma: no cover
            raise np.linalg.LinAlgError("Tanimoto Gram matrix was not factorisable")

        self.lml_, self.eta_, self.scale_, self._L, v = best
        # Only the winning eta needs its back-substitution.
        self._alpha = solve_triangular(self._L.T, v, lower=False, check_finite=False)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Posterior mean (and sd) on the original target scale."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("TanimotoGP.predict called before fit")
        if self.X_ is None:
            raise RuntimeError(
                "TanimotoGP.predict needs the training fingerprints; this model was "
                "fitted from a Gram matrix, so use predict_gram"
            )
        X = np.asarray(X, dtype=np.float32)
        Ks = tanimoto_gram(X, self.X_)
        # Prior variance of the Tanimoto kernel is k(x,x) = 1 for any non-empty
        # fingerprint, 0 for an all-zero one.
        k_diag = np.where(np.einsum("ij,ij->i", X, X) > 0, 1.0, 0.0)
        return self.predict_gram(Ks, k_diag=k_diag, return_std=return_std)

    def predict_gram(
        self, Ks: np.ndarray, k_diag: np.ndarray | None = None, return_std: bool = True
    ):
        """Posterior from a precomputed ``(n_candidates, n_train)`` Gram block."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("TanimotoGP.predict called before fit")
        Ks = np.asarray(Ks, dtype=np.float64)
        mu = Ks @ self._alpha
        mu = mu * self.y_scale_ + self.mean_
        if not return_std:
            return mu

        V = solve_triangular(self._L, Ks.T, lower=True, check_finite=False)
        diag = (np.ones(Ks.shape[0]) if k_diag is None else np.asarray(k_diag, float))
        var = np.clip(diag + self.jitter - np.einsum("ij,ij->j", V, V), 0.0, None)
        sd = np.sqrt(var * self.scale_) * self.y_scale_
        return mu, sd

    def sample_posterior(
        self, X: np.ndarray, rng: np.random.Generator, n_samples: int = 1
    ) -> np.ndarray:
        """Draw ``n_samples`` from the *marginal* posterior at each row of ``X``.

        Marginal, not joint: Thompson sampling over a discrete library only
        needs the argmax of one draw, and a joint draw over a 2000-compound
        library is a 2000x2000 Cholesky per BO step for no measurable
        difference in which molecule gets picked.
        """
        mu, sd = self.predict(X, return_std=True)
        noise = rng.standard_normal((n_samples, mu.size))
        return mu[None, :] + noise * sd[None, :]


TREND_VARIANCE_GRID: tuple[float, ...] = (
    0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
"""Trend-coefficient prior variances scanned by :class:`BasisTransferGP`.

``0.0`` is included for the same reason ``TASK_RHO_GRID`` includes it: at that
value the auxiliary basis contributes nothing and the model is exactly
:class:`TanimotoGP` on the primary observations, so the likelihood can always
decline the transfer. A gain over single-task BO is then a gain the marginal
likelihood chose, not one the model family imposed.

The upper end is deliberately generous. ``tau2 = 10`` on standardised basis
columns is a near-flat prior on the coefficients -- effectively ordinary
universal kriging -- so a family whose auxiliary targets really are a linear
predictor of the primary can say so. An interior selection is a measurement; a
selection pinned at either endpoint should be read as "the grid ran out" and is
reported by :meth:`BasisTransferGP.parameter_snapshot`."""


TASK_RHO_GRID: tuple[float, ...] = (
    0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99, 0.995, 0.999, 1.0)
"""Task correlations scanned by :class:`CoregionalisedGP`.

Both endpoints are included on purpose. ``0.0`` is exactly the
independent-target model, so the scan can always fall back to it: a multi-task
surrogate that loses to a single-task one loses because the likelihood preferred
coupling and the coupling did not pay, not because sharing was imposed on it.

``1.0`` matters for the opposite reason. An earlier version of this grid stopped
at ``0.97`` and every fit on DOCKSTRING selected exactly ``0.97`` -- which reads
like an estimate and is in fact a wall. Extending the grid moved every one of
them to ``1.0``. A selected value at the edge of its own grid is not a
measurement, and a family that cannot express "these tasks are one function"
cannot report that it found one."""


@dataclass
class CoregionalisedGP:
    """Intrinsic coregionalisation over the Tanimoto kernel.

    ``k((m, a), (m', b)) = k_tan(m, m') * B[a, b]``, with the task covariance
    restricted to the one-parameter family ``B = (1 - r) I + r 11'``.  That form
    is PSD for ``r`` in ``[0, 1]`` and has nothing to fit per task pair, so a
    campaign with four tasks and twenty observations is not being asked to
    estimate a 4x4 covariance from twenty points.  ``r`` is chosen by the exact
    log marginal likelihood -- a grid with :class:`TanimotoGP`'s analytic inner
    solve underneath, so there is no optimiser seed and no restarts, and two
    runs of the same campaign agree bit for bit.

    Each task carries its own molecule index.  That is deliberate: whether every
    task is measured on the *same* molecules (a blocked design) or on molecules
    of its own (a spread design) is the central question in allocating a
    multi-task budget, and at high task correlation the two differ by more than
    the choice of model.  It is a property of the experiment, so it does not
    belong inside the surrogate.

    Targets are z-scored per task from the observed rows only; the posterior
    mean comes back on that scale, which is what a rank correlation or an
    acquisition ordering needs.
    """

    rho_grid: tuple[float, ...] = TASK_RHO_GRID
    fixed_scale: float | None = None
    """Passed to the inner GP; see :attr:`TanimotoGP.fixed_scale`."""

    eta_grid: tuple[float, ...] | None = None
    """Noise-to-signal ratios for the inner GP. ``None`` uses
    :class:`TanimotoGP`'s own grid, which is the default and refits the ratio on
    the *joint* data. Pinning it to a single value -- typically the one a
    single-task fit chose -- holds the primary task's effective smoothing fixed
    while auxiliary rows are added, which is what separates "more rows" from
    "the extra rows changed the hyperparameters" when attributing a gain."""

    structure: str = "shared"
    """``shared``: one ``r`` for every pair, ``B = (1-r)I + r11'``.

    ``free``: a full correlation matrix, each pair fitted separately. One shared
    ``r`` is a good summary only when a family is uniformly coupled. The JAK
    block runs from 0.24 (JAK3-TYK2) to 0.71 (JAK1-JAK2), and there no single
    value is right for any pair; the CYP block spans 0.52-0.74, and there one
    value describes everything. So the free form should pay on the first and not
    the second, which makes it a prediction rather than an upgrade."""

    passes: int = 2
    """Coordinate-ascent sweeps over the pairs, for ``structure="free"``."""

    extra_diag: np.ndarray | None = None
    """Per-observation noise added to the joint kernel diagonal, in joint row
    order (task 0's rows first, then task 1's, and so on).

    ``None`` adds nothing and is the default, so every existing fit is
    unchanged bit for bit. :class:`SelectionAwareGP` uses it to down-weight
    auxiliary rows whose compounds were likely to have been chosen *because*
    of their values; see that class for why that is the right channel."""

    def _lml(self, Kb, y_j, C, idx_per_task):
        """Log marginal likelihood of a candidate task correlation, or -inf.

        Rejects a non-PSD candidate rather than repairing it: coordinate ascent
        on correlations can leave the cone, and a silently-projected matrix is
        no longer the thing whose likelihood is being compared.

        Semi-definite is allowed, and the tolerance is on the negative side
        only. ``r = 1`` gives ``B = 11'``, whose eigenvalues are ``(T, 0, ...)``
        -- singular, but a legitimate endpoint of the family and the one the
        pooling control is defined as. Demanding strict positivity rejected it
        and took the control down with it.
        """
        if np.linalg.eigvalsh(C).min() < -1e-8:
            return -np.inf, None
        T = len(idx_per_task)
        Kj = np.block([[C[a, b] * Kb[a][b] for b in range(T)] for a in range(T)])
        if self.extra_diag is not None:
            d = np.asarray(self.extra_diag, dtype=np.float64)
            if d.shape != (Kj.shape[0],):
                raise ValueError(
                    f"extra_diag has {d.shape} entries but the joint kernel is "
                    f"{Kj.shape[0]}x{Kj.shape[0]}")
            if np.any(d < 0):
                raise ValueError("extra_diag must be non-negative; a negative "
                                 "entry can take the joint kernel out of the "
                                 "PSD cone")
            Kj = Kj + np.diag(d)
        try:
            kw = {} if self.eta_grid is None else {"eta_grid": tuple(self.eta_grid)}
            if self.fixed_scale is not None:
                kw["fixed_scale"] = self.fixed_scale
            inner = TanimotoGP(**kw)
            gp = inner.fit_gram(Kj, y_j)
        except np.linalg.LinAlgError:  # pragma: no cover - jitter covers this
            return -np.inf, None
        return gp.lml_, gp

    def fit_gram(self, K: np.ndarray, idx_per_task, y_per_task) -> "CoregionalisedGP":
        """Fit from the full library Gram and per-task ``(index, values)``.

        ``idx_per_task[0]`` is the primary task: the one :meth:`predict_gram`
        returns. Tasks may share molecules or not, and may differ in size.
        """
        if len(idx_per_task) != len(y_per_task):
            raise ValueError(
                f"{len(idx_per_task)} task indices but {len(y_per_task)} value arrays")
        if not idx_per_task:
            raise ValueError("a coregionalised GP needs at least one task")

        K = np.asarray(K, dtype=np.float64)
        self.idx_ = [np.asarray(i, dtype=np.intp) for i in idx_per_task]
        z = []
        self.task_means_ = []
        self.task_scales_ = []
        for i, (idx, y) in enumerate(zip(self.idx_, y_per_task)):
            y = np.asarray(y, dtype=np.float64).ravel()
            if y.shape[0] != idx.shape[0]:
                raise ValueError(
                    f"task {i} has {idx.shape[0]} indices but {y.shape[0]} values")
            sd = float(y.std())
            sd = sd if sd > 1e-12 else 1.0
            if i == 0:
                # Keep the primary task's transform so predictions can come back
                # on the scale the caller supplied. A rank correlation does not
                # care, but an acquisition function does: expected improvement
                # compares the posterior mean against an incumbent in the
                # original units, and a z-scored mean makes every candidate look
                # equally bad, so the loop degenerates to its tie-break jitter.
                self.y0_mean_, self.y0_scale_ = float(y.mean()), sd
            self.task_means_.append(float(y.mean()))
            self.task_scales_.append(sd)
            z.append((y - y.mean()) / sd)

        T = len(self.idx_)
        y_j = np.concatenate(z)
        # The kernel blocks do not depend on B, so slice them once: the search
        # below evaluates dozens of candidate task covariances against them.
        Kb = [[K[np.ix_(self.idx_[a], self.idx_[b])] for b in range(T)]
              for a in range(T)]

        best = None
        for r in self.rho_grid:
            B = (1.0 - r) * np.eye(T) + r * np.ones((T, T))
            lml, gp = self._lml(Kb, y_j, B, self.idx_)
            if gp is not None and (best is None or lml > best[0]):
                best = (lml, r, gp, B)
        if best is None:  # pragma: no cover
            raise np.linalg.LinAlgError("no task covariance was factorisable")

        self.lml_, self.rho_, self._gp, self._B = best

        if self.structure == "free" and T > 2:
            # Coordinate ascent, from two starts. Starting only at the shared-r
            # optimum silently fails when that optimum is near 1: a correlation
            # matrix with two entries at 0.999 cannot have a third near 0 and
            # stay PSD, so every single-coordinate move is rejected and the
            # matrix freezes at the corner it began in. The independent start
            # has room to move in every coordinate; whichever run ends higher
            # wins. Both are deterministic, so two runs still agree bit for bit.
            starts = [self._B.copy(), np.eye(T)]
            for C0 in starts:
                C, lml_c = C0.copy(), self._lml(Kb, y_j, C0, self.idx_)[0]
                gp_c = None
                for _ in range(self.passes):
                    moved = False
                    for a in range(T):
                        for b in range(a + 1, T):
                            for c in self.rho_grid:
                                if c == C[a, b]:
                                    continue
                                cand = C.copy()
                                cand[a, b] = cand[b, a] = c
                                lml, gp = self._lml(Kb, y_j, cand, self.idx_)
                                if gp is not None and lml > lml_c + 1e-9:
                                    lml_c, gp_c, C = lml, gp, cand
                                    moved = True
                    if not moved:
                        break
                if gp_c is not None and lml_c > self.lml_ + 1e-9:
                    self.lml_, self._gp, self._B = lml_c, gp_c, C
        return self

    def parameter_snapshot(self) -> dict:
        """Serializable parameters and transforms of an already fitted donor."""
        if not hasattr(self, "_gp"):
            raise RuntimeError("parameter_snapshot requires a fitted model")
        return {
            "rho": float(self.rho_), "task_covariance": self._B.tolist(),
            "eta": float(self._gp.eta_), "signal_variance": float(self._gp.scale_),
            "jitter": float(self._gp.jitter),
            "task_means": list(self.task_means_),
            "task_scales": list(self.task_scales_),
            "inner_mean": float(self._gp.mean_),
            "inner_response_scale": float(self._gp.y_scale_),
            "primary_mean": float(self.y0_mean_),
            "primary_scale": float(self.y0_scale_),
        }

    def recondition(self, y_per_task) -> "CoregionalisedGP":
        """Condition on replacement responses at this donor's locations.

        No parameters or normalization transforms are re-estimated. The donor's
        Cholesky factor is reused, so predictive variance is unchanged. The
        returned object has its own response coefficients and does not mutate
        the donor. This is a fixed-parameter posterior, not a refitted model.
        """
        if not hasattr(self, "_gp"):
            raise RuntimeError("recondition requires a fitted donor")
        if len(y_per_task) != len(self.idx_):
            raise ValueError("replacement responses must match the donor's tasks")
        normalized = []
        for idx, y, mean, scale in zip(self.idx_, y_per_task,
                                      self.task_means_, self.task_scales_):
            y = np.asarray(y, dtype=np.float64).ravel()
            if len(y) != len(idx) or not np.all(np.isfinite(y)):
                raise ValueError("replacement responses must be finite and match donor locations")
            normalized.append((y - mean) / scale)
        z = (np.concatenate(normalized) - self._gp.mean_) / self._gp.y_scale_
        out = copy(self)
        out._gp = copy(self._gp)
        v = solve_triangular(self._gp._L, z, lower=True, check_finite=False)
        out._gp._alpha = solve_triangular(self._gp._L.T, v, lower=False,
                                          check_finite=False)
        # lml_ remains the donor fit's likelihood; no refit is performed here.
        return out

    def predict_gram(self, K: np.ndarray, candidates, return_std: bool = False):
        """Posterior on the primary task, for rows ``candidates`` of ``K``.

        With ``return_std``, also the posterior sd -- which is what an
        acquisition function needs, and the reason a multi-task surrogate can
        be dropped into a BO loop at all. The prior variance of the primary
        task is ``B[0, 0] * k(x, x) = 1`` for any parseable fingerprint, since
        the task correlation matrix carries a unit diagonal by construction.
        """
        if not hasattr(self, "_gp"):
            raise RuntimeError("CoregionalisedGP.predict_gram called before fit_gram")
        c = np.asarray(candidates, dtype=np.intp)
        Ks = np.hstack([self._B[0, b] * np.asarray(K)[np.ix_(c, self.idx_[b])]
                        for b in range(len(self.idx_))])
        if not return_std:
            mu = self._gp.predict_gram(Ks, return_std=False)
            return mu * self.y0_scale_ + self.y0_mean_
        mu, sd = self._gp.predict_gram(Ks, k_diag=np.ones(len(c)), return_std=True)
        return mu * self.y0_scale_ + self.y0_mean_, sd * self.y0_scale_


def loo_log_predictive(gp: "TanimotoGP", rows=None) -> float:
    """Leave-one-out log predictive density of a fitted :class:`TanimotoGP`.

    Closed form, no refitting: for ``z ~ N(0, C)`` the LOO predictive at row
    ``i`` has variance ``1 / [C^-1]_ii`` and residual ``[C^-1 z]_i / [C^-1]_ii``
    (Rasmussen & Williams, *Gaussian Processes for Machine Learning*, eqs.
    5.10-5.12). Here ``C = scale * (K0 + eta I)``, whose Cholesky factor and
    ``alpha = (K0 + eta I)^-1 z`` the fit already stored, so the whole quantity
    costs one triangular solve.

    ``rows`` restricts the sum, which is the point of having this: a
    multi-task fit can be scored on the primary task's rows alone.

    The density is of the *inner standardised* target. The standardising
    transform does not depend on the parameters being compared, so the
    Jacobian is a shared constant and is omitted.
    """
    if not getattr(gp, "_fitted", False):
        raise RuntimeError("loo_log_predictive needs a fitted TanimotoGP")
    L = gp._L
    n = L.shape[0]
    # diag((K0 + eta I)^-1) = column norms of L^-1, since M^-1 = L^-T L^-1.
    Linv = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
    minv_diag = np.einsum("ij,ij->j", Linv, Linv)
    var = gp.scale_ / minv_diag
    resid = gp._alpha / minv_diag
    lpd = -0.5 * (np.log(2.0 * np.pi * var) + resid ** 2 / var)
    if rows is not None:
        lpd = lpd[np.asarray(rows, dtype=np.intp)]
    return float(lpd.sum())


SELECTION_GAMMA_GRID: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
"""Strength of the selection correction in :class:`SelectionAwareGP`.

``0.0`` is the uncorrected coregionalised model, reached exactly rather than in
the limit, so the correction can be *declined* by the likelihood and the arm
reproduces :class:`CoregionalisedGP` bit for bit when it is. The top of the
grid is far past the useful range for the same reason ``TASK_RHO_GRID`` runs to
1.0: a value selected at the edge of its own grid is a wall, not an estimate.
"""


@dataclass
class SelectionAwareGP:
    """Coregionalisation that discounts auxiliary rows by how predictable the
    panel's membership was.

    **The problem.** A coregionalised GP conditions on auxiliary measurements
    as though the compounds carrying them were an exchangeable sample of the
    library. They are not. Panels are assembled by people, and
    ``scripts/panel_propensity.py`` measures the consequence in BindingDB: the
    compounds co-assayed across a family are enriched for actives in some
    families (JAK2 holds 10 of its top 10) and depleted in others (CYP3A4
    holds 0 of 4.5 expected). Conditioning on a selected observation as if it
    were exchangeable overstates how much it says, and in this model the
    overstatement lands on the *variance*: the predictive sd at every
    panel-covered compound is deflated, expected improvement reads the
    deflation as "already explored", and the campaign steers away from exactly
    the compounds the panel covers. That channel, not the prior mean, carries
    the design artefact the crossed controls isolate.

    **The correction.** Estimate the selection propensity ``pi(x) = P(x is in
    the panel | structure)`` and inflate the auxiliary observation noise in
    proportion to it::

        noise_i = base + gamma * (pi_i - min pi)      (auxiliary rows only)

    An observation weight of ``1 / pi`` is the Horvitz-Thompson weight
    (Horvitz & Thompson, JASA 47(260):663-685, 1952), and a weight enters a
    Gaussian likelihood as a reciprocal noise variance, so noise proportional
    to ``pi`` *is* inverse-propensity weighting written in kernel form. The
    shift by ``min pi`` makes the term non-negative, which keeps the joint
    kernel PSD; a constant offset would be absorbed by ``eta`` anyway, so only
    the *differential* part of the propensity is identified, which is the part
    that encodes selection.

    ``gamma`` is chosen by the same exact marginal likelihood that chooses
    ``r`` and ``eta``. The model is therefore free to decline the correction,
    and at ``gamma = 0`` it is :class:`CoregionalisedGP` exactly.

    **Where the idea comes from.** Three fields solved this problem before
    molecular BO met it, and the construction above borrows from each:

    * *Geostatistics.* Diggle, Menezes & Su (JRSS-C 59(2):191-232, 2010) named
      preferential sampling and modelled the locations and the field jointly.
      Closer still, Schliep, Wikle & Daw (arXiv:2108.12354) correct spatial
      covariance parameters *and the kriging variance* under informative
      sampling by weighted composite likelihood -- the same quantity and the
      same device as here.
    * *Recommender systems.* Ratings are missing not at random because people
      choose what to watch, and Schnabel et al. (ICML 2016) debias matrix
      completion by inverse propensity. Their standing difficulty is that
      estimating the propensity needs a missing-completely-at-random sample
      that production logs do not contain. **A dense co-measured block is
      exactly that sample**, which is why this correction can be validated
      here and not there.
    * *Ecology.* Fithian, Elith, Hastie & Keith (Methods Ecol Evol, 2015)
      estimate presence-only sampling bias by pooling species and assuming the
      bias is shared, borrowing strength across them. The propensity here is
      likewise shared across a family's targets rather than fitted per target.

    **What is not claimed.** The propensity is estimated from co-assay, which
    is observational: a compound may sit in a panel because somebody expected
    it to be potent, or because one paper ran one series against several
    targets. This corrects the association, and does not identify the
    mechanism.
    """

    rho_grid: tuple[float, ...] = TASK_RHO_GRID
    gamma_grid: tuple[float, ...] = SELECTION_GAMMA_GRID
    fixed_scale: float | None = None
    eta_grid: tuple[float, ...] | None = None
    structure: str = "shared"
    gamma_: float = field(default=0.0, repr=False)
    propensity_: np.ndarray | None = field(default=None, repr=False)
    gamma_loo_: tuple[float, ...] = ()
    """Primary-task LOO log predictive density at each grid value of ``gamma``,
    in grid order. Reported so a selection at the edge of the grid is visible
    as the wall it is."""

    @staticmethod
    def propensity(K, selected, n_library, *, folds: int = 5):
        """Out-of-fold estimate of ``P(compound is in the panel | structure)``.

        A membership indicator regressed on the Tanimoto Gram, through the same
        :class:`TanimotoGP` the rest of the module uses, so the smoothing is
        chosen by the exact likelihood and there is no optimiser to seed.

        The estimate is **out of fold**. Fitted in-sample, a GP with a small
        fitted ``eta`` interpolates the indicator, the propensity collapses to
        the indicator itself, the correction becomes a constant on the
        auxiliary rows, and ``eta`` absorbs it -- a silent no-op that still
        reports a fitted ``gamma``. Out-of-fold prediction also makes the
        estimate honest enough to score: its AUC says whether panel membership
        was predictable from structure at all, which is the quantity Pogodin
        et al. (2019) report at roughly 0.87 for ChEMBL.

        Reads no response value, primary or auxiliary -- only *which* compounds
        the panel covers.

        **Shrunk by its own measured skill.** Out-of-fold prediction stops the
        estimate being *right* about random labels; it does not stop it being
        *variable* about them. Measured on a shuffled panel the raw estimate
        still had a spread of 0.155 against 0.188 for the real one, so a
        correction driven by it fires just as hard on noise as on selection --
        the identifiability failure Copas selection models are known for in
        meta-analysis. So the estimate is shrunk toward the base rate by its
        out-of-fold coefficient of determination::

            s  = max(0, 1 - SS_res / SS_tot)
            pi = base + s * (pi_raw - base)

        A propensity with no out-of-fold skill has ``s = 0``, collapses to a
        constant, and leaves the model exactly uncorrected. No new
        hyperparameter: ``s`` is measured, not chosen.
        """
        K = np.asarray(K, dtype=np.float64)
        ind = np.zeros(int(n_library), dtype=np.float64)
        ind[np.asarray(selected, dtype=np.intp)] = 1.0
        base = float(ind.mean())
        if ind.all() or not ind.any():
            # Nothing to learn: every compound is in the panel, or none is.
            return np.full(len(ind), base)
        pi = np.empty(len(ind), dtype=np.float64)
        fold = np.arange(len(ind)) % int(folds)
        for f in range(int(folds)):
            te = fold == f
            tr = ~te
            if not tr.any() or not te.any() or ind[tr].std() < 1e-12:
                pi[te] = float(ind[tr].mean()) if tr.any() else base
                continue
            gp = TanimotoGP().fit_gram(K[np.ix_(tr, tr)], ind[tr])
            pi[te] = gp.predict_gram(K[np.ix_(te, tr)], return_std=False)
        pi = np.clip(pi, 0.0, 1.0)
        ss_tot = float(((ind - base) ** 2).sum())
        ss_res = float(((ind - pi) ** 2).sum())
        skill = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
        return np.clip(base + skill * (pi - base), 0.0, 1.0)

    def fit_gram(self, K, idx_per_task, y_per_task,
                 propensity=None) -> "SelectionAwareGP":
        """Fit, scanning ``gamma`` against the exact joint likelihood.

        ``propensity`` is over the whole library. ``None`` means no correction
        is available, and the fit is then :class:`CoregionalisedGP` exactly.
        """
        idx = [np.asarray(i, dtype=np.intp) for i in idx_per_task]
        sizes = [len(i) for i in idx]
        n_joint = int(sum(sizes))

        if propensity is None:
            grid, pi = (0.0,), None
        else:
            pi = np.asarray(propensity, dtype=np.float64)
            # Validate here rather than relying on the guard in
            # CoregionalisedGP._lml. That guard sees `gamma * (pi - min pi)`,
            # which the shift makes non-negative for *any* input, so an
            # out-of-range propensity would otherwise be silently rescaled
            # into something plausible instead of rejected.
            if pi.ndim != 1:
                raise ValueError(f"propensity must be 1-d, got shape {pi.shape}")
            if not np.all(np.isfinite(pi)):
                raise ValueError("propensity contains non-finite values")
            if pi.min() < 0.0 or pi.max() > 1.0:
                raise ValueError(
                    f"propensity must lie in [0, 1]; got "
                    f"[{pi.min():.3g}, {pi.max():.3g}]")
            need = max((int(i.max()) for i in idx if len(i)), default=-1) + 1
            if len(pi) < need:
                raise ValueError(
                    f"propensity has {len(pi)} entries but the task indices "
                    f"reach library row {need - 1}")
            grid = tuple(self.gamma_grid)
            # Only auxiliary rows are corrected. The primary task's own
            # observations are the campaign's picks, not the panel's, and the
            # acquisition function that chose them is already accounted for.
            aux = np.concatenate([pi[i] for i in idx[1:]]) if len(idx) > 1 \
                else np.empty(0)
            lo = float(aux.min()) if aux.size else 0.0

        # gamma is scored by leave-one-out predictive density on the PRIMARY
        # rows, not by the joint marginal likelihood. The joint likelihood is
        # rewarded for explaining the auxiliary observations, which is exactly
        # what this correction sets out to discount, so it drives gamma to
        # zero whatever the selection looks like: measured on the synthetic
        # check it moved the panel sd ratio only 0.645 -> 0.679. Scoring the
        # estimand the campaign actually uses is the same move Hadad et al.
        # (PNAS 118(15), 2021) make for adaptively collected data. Every other
        # parameter is still fitted by the exact joint likelihood inside
        # CoregionalisedGP; only this one trades auxiliary fit against primary
        # prediction, and only this one therefore needs a held-out criterion.
        primary_rows = np.arange(sizes[0], dtype=np.intp)
        best, scores = None, []
        for gamma in grid:
            if pi is None or gamma == 0.0:
                extra = None
            else:
                extra = np.zeros(n_joint, dtype=np.float64)
                at = sizes[0]
                for i in idx[1:]:
                    extra[at:at + len(i)] = gamma * (pi[i] - lo)
                    at += len(i)
            co = CoregionalisedGP(
                rho_grid=self.rho_grid, fixed_scale=self.fixed_scale,
                eta_grid=self.eta_grid, structure=self.structure,
                extra_diag=extra)
            try:
                co.fit_gram(K, idx, y_per_task)
                score = loo_log_predictive(co._gp, primary_rows)
            except np.linalg.LinAlgError:  # pragma: no cover
                scores.append(float("-inf"))
                continue
            if not np.isfinite(score):  # pragma: no cover
                scores.append(float("-inf"))
                continue
            scores.append(score)
            # Ties go to the smaller gamma: the grid is ascending, so a strict
            # `>` keeps the least-corrected model that fits equally well.
            if best is None or score > best[0]:
                best = (score, gamma, co)
        if best is None:  # pragma: no cover
            raise np.linalg.LinAlgError("no selection correction was factorisable")

        self.loo_, self.gamma_, self._co = best
        self.lml_ = float(self._co.lml_)
        self.gamma_loo_ = tuple(scores)
        self.propensity_ = pi
        self.rho_ = self._co.rho_
        self.idx_ = self._co.idx_
        return self

    def predict_gram(self, K, candidates, return_std: bool = False):
        """Posterior on the primary task; see :meth:`CoregionalisedGP.predict_gram`.

        The correction is observation noise, so it changes the training
        covariance only. The prior variance at a candidate is untouched, which
        is the point: a compound the panel covers keeps its honest predictive
        sd instead of the deflated one.
        """
        if not hasattr(self, "_co"):
            raise RuntimeError("SelectionAwareGP.predict_gram called before fit_gram")
        return self._co.predict_gram(K, candidates, return_std=return_std)


@dataclass
class BasisTransferGP:
    """Transfer through the prior *mean* only: auxiliary assays as basis functions.

    The coregionalised model transfers through the covariance, which is why an
    auxiliary measurement at compound ``m`` shrinks the primary-target
    predictive variance at ``m``. Section 40 measured that shrinkage (0.412
    against 0.542 on CYP) and section 50 showed what it costs: expected
    improvement reads the compressed variance as "already explored" and the
    pooled arm selects historically assayed compounds at 0.288 of their library
    share, paying a value-free selection penalty in every family. The penalty
    is paid whether or not the auxiliary responses carry any information, which
    is what makes it a modelling artefact rather than a finding.

    This model removes that channel by construction rather than by correction.
    Each auxiliary target is fitted on its own data, and its posterior mean over
    the library becomes a *basis function* for the primary task::

        f(x) = h(x)' beta + g(x),    g ~ GP(0, k_tan),    beta ~ N(0, tau2 I)

    with ``h(x)`` the standardised auxiliary posterior means. Marginalising over
    ``beta`` leaves a Gaussian process with the augmented kernel ``k_tan(x, x')
    + tau2 h(x)' h(x')`` (Rasmussen & Williams, Gaussian Processes for Machine
    Learning, section 2.7), so the whole model is one more Tanimoto GP and
    inherits this module's exact likelihood, its eta scan and its determinism.

    Three properties follow, and each is pinned by a test:

    **Historical compounds are not made artificially certain.** The auxiliary
    rows are not entries in the conditioning set: they reach the posterior only
    through a ``T``-dimensional trend, so whatever variance they remove, they
    remove everywhere the trend is confident rather than specifically at the
    compounds that happen to have been assayed. Nine hundred historical rows
    shrink the posterior through three fitted coefficients, not through nine
    hundred conditioning events. The readable consequence is the statistic
    section 50 reports -- mean predictive sd at historically assayed compounds
    over mean sd elsewhere. On synthetic data from this kernel that ratio is
    0.99 for ``stl``, **0.51** for the coregionalised model, and **1.05** here.

    Note what is *not* claimed. The basis is built from the auxiliary values,
    and ``tau2`` is refitted against them, so replacing those values with noise
    does move the predictive sd -- it should, and it does so in the right
    direction (the likelihood drops ``tau2`` from 1.0 to 0.03 on a noise
    basis). The property is the absence of *location-specific* deflation, which
    is the thing that drives expected improvement away from assayed compounds.
    A test pins the ratio; none pins bit-identity, because bit-identity is
    false.

    **Each auxiliary target gets its own fitted coefficient.** ``beta_`` is one
    number per auxiliary assay, estimated by the same marginal likelihood that
    chooses everything else here, and it answers the inclusion question
    directly: a coefficient at zero is an assay the model declined to use. No
    threshold is fitted across blocks, so nothing here repeats section 44's
    failure -- the decision is made inside a single campaign from that
    campaign's own primary observations.

    **It is cheaper than the model it replaces.** The auxiliary fits happen once
    (history does not grow during a campaign) and cost ``O(sum_j n_j^3)``; each
    BO round is then ``O(n_t^3)`` in the primary observations alone, against
    ``O((n_t + sum_j n_j)^3)`` per round for the coregionalised GP. On a
    four-target block with 300 historical rows each that is a 100x100 Cholesky
    per round instead of a 1000x1000 one.

    What it gives up is real and should not be glossed: a genuinely
    well-specified coregionalised model with a correctly estimated task
    correlation *should* borrow variance, and refusing to do so is a loss of
    statistical efficiency. Tighineanu et al. (arXiv:2111.11223) argue exactly
    this against mean-only transfer, on synthetic benchmarks. Whether the
    efficiency loss or the selection artefact dominates is an empirical
    question about the data, which is why this is an arm and not a replacement.

    Construct with :meth:`from_auxiliary`, which does the history fitting once,
    then call :meth:`fit_gram` per round with the primary observations.
    """

    tau2_grid: tuple[float, ...] = TREND_VARIANCE_GRID
    eta_grid: tuple[float, ...] = ETA_GRID
    fixed_scale: float | None = None
    """Passed to the inner GP; see :attr:`TanimotoGP.fixed_scale`."""

    rank: int | None = None
    """Truncate the basis to its leading ``rank`` principal directions.

    ``None`` keeps one column per auxiliary target. An integer replaces them
    with the leading left singular vectors of the stacked columns, which is
    worth doing whenever the auxiliary targets are tightly coupled: on CYP the
    pairwise correlations run 0.52-0.74, so three raw columns span barely more
    than one direction and the coefficient prior is asked to resolve a nearly
    degenerate basis. Orthogonalising removes that, and truncating is a
    regulariser with a chemical reading -- the leading direction is general
    activity against the family, and the later ones are what separates one
    target from another.

    The decomposition runs on the auxiliary *posterior means*, which are
    complete over the library by construction. Running it on the raw history
    matrix instead would require imputing the unmeasured entries, and the
    imputation would put structure into the basis that no assay measured.

    Directions are whitened to unit variance rather than kept at their singular
    values, so ``tau2`` is one prior variance per orthogonal direction and the
    likelihood, not the auxiliary spectrum, decides which directions earn
    weight."""

    H_: np.ndarray | None = field(default=None, repr=False)
    """``(n_library, n_aux)`` standardised auxiliary posterior means."""

    aux_lml_: tuple[float, ...] = ()
    """Log marginal likelihood of each auxiliary fit, for diagnostics."""

    @classmethod
    def from_auxiliary(cls, K, aux_idx, aux_val, *, n_library=None, **kwargs):
        """Fit one GP per auxiliary target and cache the basis over the library.

        Separated from :meth:`fit_gram` because the history is fixed for the
        whole campaign: refitting it every round would multiply the cost of the
        arm by the number of rounds and could not change the answer. ``K`` is
        the full library Gram.

        Each column is standardised over the library, so ``tau2`` means the same
        thing for an auxiliary target whose assay happens to span three log
        units as for one that spans one. No primary-target value is touched
        here, which is what keeps the basis free of leakage.
        """
        K = np.asarray(K, dtype=np.float64)
        n = K.shape[0] if n_library is None else int(n_library)
        self = cls(**kwargs)
        cols, lmls = [], []
        for idx, val in zip(aux_idx, aux_val):
            idx = np.asarray(idx, dtype=np.intp)
            val = np.asarray(val, dtype=np.float64).ravel()
            if idx.size == 0:
                continue
            if idx.shape[0] != val.shape[0]:
                raise ValueError(
                    f"auxiliary task has {idx.shape[0]} indices but {val.shape[0]} values")
            gp = TanimotoGP(eta_grid=self.eta_grid).fit_gram(
                K[np.ix_(idx, idx)], val)
            m = gp.predict_gram(K[:, idx], return_std=False)
            sd = float(m.std())
            # A flat column carries no information and would divide by zero on
            # standardisation; drop it rather than letting a numerical artefact
            # become a basis function the likelihood then has to shrink away.
            if sd <= 1e-12:
                continue
            cols.append((m - m.mean()) / sd)
            lmls.append(float(gp.lml_))
        H = (np.column_stack(cols) if cols
             else np.zeros((n, 0), dtype=np.float64))
        self.H_ = self._reduce(H, self.rank)
        self.aux_lml_ = tuple(lmls)
        return self

    @classmethod
    def from_chemistry(cls, K, *, rank: int = 3, **kwargs):
        """A rank-matched basis built from the library's chemistry alone.

        The control for the method, not the method. It takes no auxiliary
        argument at all -- not the values, not the locations -- so "this basis
        carries no auxiliary information" is a property of the signature rather
        than something a test has to establish. If an arm built on this
        reproduces the gain that :meth:`from_auxiliary` shows, then the
        auxiliary assays are not what is working and the trend is reading
        library structure the Tanimoto kernel already had.

        The matched null in between -- same history locations, same response
        distribution, correspondence destroyed -- is the existing
        ``--aux-shuffle`` path applied to :meth:`from_auxiliary`, so the three
        together bracket the claim: chemistry only, locations only, and the
        real thing.
        """
        K = np.asarray(K, dtype=np.float64)
        # Kernel PCA: eigenvectors of the centred Gram, largest first.
        n = K.shape[0]
        J = np.eye(n) - np.full((n, n), 1.0 / n)
        w, V = np.linalg.eigh(J @ K @ J)
        H = V[:, ::-1][:, :max(int(rank), 0)]
        self = cls(rank=None, **kwargs)
        self.H_ = self._reduce(H, None)
        self.aux_lml_ = ()
        return self

    @staticmethod
    def _reduce(H, rank):
        """Standardise the basis, optionally rotating to its leading directions."""
        H = np.asarray(H, dtype=np.float64)
        if H.shape[1] == 0:
            return H
        if rank is not None:
            # Left singular vectors of the centred basis. `full_matrices=False`
            # keeps this an (n x T) decomposition rather than an (n x n) one.
            #
            # Applied whenever a rank is given, including `rank == n_columns`.
            # At full rank this is a pure rotation and changes nothing a linear
            # model could express, but it is still worth doing: the raw columns
            # of a tightly coupled family are near-collinear, and an orthogonal
            # basis is what makes a single `tau2` mean the same thing in every
            # direction. `None` is the way to ask for no rotation at all.
            U, _, _ = np.linalg.svd(H - H.mean(axis=0), full_matrices=False)
            H = U[:, :max(int(rank), 0)]
        sd = H.std(axis=0)
        keep = sd > 1e-12
        H = H[:, keep]
        if H.shape[1] == 0:
            return np.zeros((len(H), 0), dtype=np.float64)
        return (H - H.mean(axis=0)) / H.std(axis=0)

    def fit_gram(self, K, obs_idx, y_obs) -> "BasisTransferGP":
        """Condition on the primary observations, scanning ``tau2`` by likelihood."""
        if self.H_ is None:
            raise RuntimeError("fit_gram requires a basis; use from_auxiliary")
        K = np.asarray(K, dtype=np.float64)
        obs = np.asarray(obs_idx, dtype=np.intp)
        y = np.asarray(y_obs, dtype=np.float64).ravel()
        if obs.shape[0] != y.shape[0]:
            raise ValueError(
                f"{obs.shape[0]} observation indices but {y.shape[0]} values")
        self.obs_ = obs
        Hobs = self.H_[obs]
        K0 = K[np.ix_(obs, obs)]

        best = None
        for tau2 in self.tau2_grid:
            # The augmented kernel IS the marginalisation over beta, so the
            # likelihood compared across the grid is the exact marginal
            # likelihood of the semi-parametric model -- not a profile
            # likelihood with beta plugged in at its mode.
            Kaug = K0 if tau2 == 0.0 else K0 + tau2 * (Hobs @ Hobs.T)
            kw = {"eta_grid": self.eta_grid}
            if self.fixed_scale is not None:
                kw["fixed_scale"] = self.fixed_scale
            try:
                gp = TanimotoGP(**kw).fit_gram(Kaug, y)
            except np.linalg.LinAlgError:  # pragma: no cover - jitter covers this
                continue
            if best is None or gp.lml_ > best[0]:
                best = (gp.lml_, tau2, gp)
        if best is None:  # pragma: no cover
            raise np.linalg.LinAlgError("no trend variance was factorisable")

        self.lml_, self.tau2_, self._gp = best
        # Posterior mean of the trend coefficients, free of charge: for the
        # linear part of an additive kernel, beta_hat = tau2 * H' alpha. In the
        # inner GP's z-units; `beta_response_` puts it back on the primary scale.
        self.beta_ = (self.tau2_ * (Hobs.T @ self._gp._alpha)
                      if Hobs.shape[1] else np.zeros(0))
        self.beta_response_ = self.beta_ * self._gp.y_scale_
        return self

    def predict_gram(self, K, candidates, return_std: bool = False):
        """Posterior on the primary task, for rows ``candidates`` of ``K``."""
        if not hasattr(self, "_gp"):
            raise RuntimeError("BasisTransferGP.predict_gram called before fit_gram")
        c = np.asarray(candidates, dtype=np.intp)
        Ks = np.asarray(K)[np.ix_(c, self.obs_)]
        Hc = self.H_[c]
        if self.tau2_ != 0.0 and Hc.shape[1]:
            Ks = Ks + self.tau2_ * (Hc @ self.H_[self.obs_].T)
        if not return_std:
            return self._gp.predict_gram(Ks, return_std=False)
        # Prior variance at x is k(x,x) + tau2 h(x)'h(x): unit Tanimoto variance
        # plus the trend's own uncertainty. The second term is what an honest
        # mean-transfer model must carry -- the basis is an estimate, and a
        # compound where the auxiliary models disagree with each other is a
        # compound this model is entitled to be less sure about.
        k_diag = np.ones(len(c))
        if self.tau2_ != 0.0 and Hc.shape[1]:
            k_diag = k_diag + self.tau2_ * np.einsum("ij,ij->i", Hc, Hc)
        return self._gp.predict_gram(Ks, k_diag=k_diag, return_std=True)

    def parameter_snapshot(self) -> dict:
        """Serializable fitted parameters, including the per-assay coefficients."""
        if not hasattr(self, "_gp"):
            raise RuntimeError("parameter_snapshot requires a fitted model")
        return {
            "tau2": float(self.tau2_),
            "tau2_at_grid_edge": bool(
                self.tau2_ in (self.tau2_grid[0], self.tau2_grid[-1])),
            "beta": [float(b) for b in self.beta_],
            "beta_response_scale": [float(b) for b in self.beta_response_],
            "n_basis": int(self.H_.shape[1]),
            "rank": (None if self.rank is None else int(self.rank)),
            "eta": float(self._gp.eta_),
            "signal_variance": float(self._gp.scale_),
            "lml": float(self.lml_),
            "auxiliary_lml": list(self.aux_lml_),
        }


class LibraryKernel:
    """The full Tanimoto Gram of a fixed library, computed once and sliced.

    A campaign over an enumerated library asks for the same similarities over
    and over: every BO round rebuilds the candidates-vs-observed block from
    scratch, and across a 20-seed x 4-arm sweep the same 2000x2000 matrix gets
    recomputed thousands of times.  Precomputing it turns the dominant cost of
    the primary tier into array indexing.

    Stored as float32 -- the Gram is only ever used to *build* the small train
    blocks, which are cast back to float64 before any Cholesky, so the
    factorisation keeps full precision while the big matrix costs half the
    memory.  A 2000-compound library is 16 MB; :meth:`maybe_build` declines
    anything that would not fit comfortably, and the caller falls back to
    computing blocks on the fly.
    """

    def __init__(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.K = tanimoto_gram(self.X, self.X).astype(np.float32)
        self.k_diag = np.where(np.einsum("ij,ij->i", self.X, self.X) > 0, 1.0, 0.0)

    @classmethod
    def maybe_build(cls, X: np.ndarray, max_bytes: int = 512 * 1024 * 1024):
        """Build the cache, or return ``None`` if the library is too large."""
        n = np.asarray(X).shape[0]
        return cls(X) if n * n * 4 <= max_bytes else None

    def train_block(self, idx) -> np.ndarray:
        i = np.asarray(idx, dtype=np.intp)
        return self.K[np.ix_(i, i)].astype(np.float64)

    def cross_block(self, cand, train) -> np.ndarray:
        c = np.asarray(cand, dtype=np.intp)
        t = np.asarray(train, dtype=np.intp)
        return self.K[np.ix_(c, t)].astype(np.float64)

    def __len__(self) -> int:
        return self.X.shape[0]


class PackedFingerprints:
    """Bit-packed ECFP for a pool too large to hold as floats.

    A 250k-molecule pool is 2 GB as float32 (250000 x 2048 x 4) and will not
    fit alongside everything else; packed to bits it is 64 MB, because that is
    what the data actually is. Rows are unpacked to float32 only for the
    candidates a round actually scores.

    This is what makes PMO's ZINC-250k pool searchable with an exact GP: the
    surrogate never sees more than the sampled candidate set at once.
    """

    def __init__(self, smiles_list, n_bits: int = 2048, progress: bool = False):
        self.n_bits = n_bits
        n = len(smiles_list)
        self.packed = np.zeros((n, (n_bits + 7) // 8), dtype=np.uint8)
        self.valid = np.zeros(n, dtype=bool)
        for i, smi in enumerate(smiles_list):
            fp = morgan_fingerprint(smi, as_numpy=True)
            if fp is not None:
                self.packed[i] = np.packbits(fp.astype(np.uint8))
                self.valid[i] = True
            if progress and i % 25_000 == 0 and i:
                print(f"    fingerprinted {i:,}/{n:,}", flush=True)

    def __len__(self) -> int:
        return self.packed.shape[0]

    def take(self, idx) -> np.ndarray:
        """``(len(idx), n_bits)`` float32 rows, unpacked on demand."""
        i = np.asarray(idx, dtype=np.intp)
        return np.unpackbits(self.packed[i], axis=1)[:, : self.n_bits].astype(np.float32)

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes

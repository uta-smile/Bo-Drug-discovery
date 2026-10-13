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

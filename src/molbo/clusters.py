"""Clustering the library, and choosing which cluster to spend budget in.

The idea under test: partition the library by structural similarity, let
something pick a region, and let BO search inside it. Restricting the
candidate pool is a *constraint*, so this rides the existing
``ArmConfig.constraint_source`` hook with ``constraint_every=1`` -- a fresh
region each round.

Two things decide whether any of this can work, and both are measurable before
a model is trained:

**Is the good stuff concentrated?** If the library's true top 1% is spread
evenly across clusters, then every region is equally good and no chooser --
learned, bandit, or oracle -- can help. :func:`concentration` answers this with
no BO at all.

**Are there few enough regions to choose between?** A bandit needs arms far
fewer than pulls. Real chemical libraries shatter: 504 cruzipain compounds fall
into 299 Murcko scaffolds, so a 200-call budget buys 0.67 pulls per arm and the
bandit never leaves its first round. Coarse k-means clusters are choosable but
chemically arbitrary; chemically coherent clusters are too numerous. That
tension is the design problem, and :func:`coherence` measures where it bites.

The baseline any learned chooser has to beat is not plain BO -- it is
:class:`UCBClusterSource`, ten lines with regret guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .constraints import Constraint, LibraryContext


# --------------------------------------------------------------------------
# partitioning
# --------------------------------------------------------------------------


def cluster_library(smiles: Sequence[str], k: int, method: str = "kmeans",
                    seed: int = 0) -> np.ndarray:
    """-> integer label per molecule.

    ``kmeans``
        MiniBatchKMeans on the fingerprint matrix. Gives ``k`` balanced,
        choosable regions, but Euclidean distance on a binary fingerprint is
        not Tanimoto, so the clusters are not chemically principled.
    ``scaffold``
        Bemis-Murcko scaffold identity, then the ``k`` largest scaffolds as
        their own clusters and everything else pooled into one. Chemically
        meaningful and what a chemist would recognise, but the tail is enormous.
    ``sphere``
        Greedy sphere exclusion on Tanimoto similarity at a threshold chosen to
        land near ``k`` clusters. Native to cheminformatics; tends to produce
        many singletons on a diverse library.
    """
    from .surrogate import fingerprint_matrix

    smiles = list(smiles)
    if method == "scaffold":
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        scaf = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else s)
        counts: dict[str, int] = {}
        for s in scaf:
            counts[s] = counts.get(s, 0) + 1
        top = {s: i for i, (s, _) in
               enumerate(sorted(counts.items(), key=lambda kv: -kv[1])[:k - 1])}
        return np.array([top.get(s, k - 1) for s in scaf])

    X = fingerprint_matrix(smiles)
    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(n_clusters=min(k, len(smiles)), n_init=3,
                               random_state=seed, batch_size=2048).fit_predict(X)
    if method == "sphere":
        from .surrogate import tanimoto_gram

        K = tanimoto_gram(X, X).astype(np.float32)
        # Bisect the similarity threshold to land near k clusters, because the
        # cluster count is wildly threshold-sensitive on a diverse library.
        lo, hi = 0.05, 0.95
        best = None
        for _ in range(12):
            thr = 0.5 * (lo + hi)
            lab = _sphere(K, thr)
            n = lab.max() + 1
            if best is None or abs(n - k) < abs(best[1] - k):
                best = (lab, n)
            if n > k:
                lo = thr          # too many clusters -> loosen
            else:
                hi = thr
        return best[0]
    raise ValueError(f"unknown method {method!r}; expected kmeans, scaffold or sphere")


def _sphere(K: np.ndarray, thr: float) -> np.ndarray:
    n = len(K)
    unassigned = np.ones(n, bool)
    lab = np.full(n, -1)
    c = 0
    for i in np.argsort(-K.sum(1)):          # densest centre first
        if not unassigned[i]:
            continue
        members = unassigned & (K[i] >= thr)
        lab[members] = c
        unassigned &= ~members
        c += 1
    return lab


# --------------------------------------------------------------------------
# is this partition worth choosing between?
# --------------------------------------------------------------------------


def concentration(labels: np.ndarray, values: np.ndarray, top_frac: float = 0.01) -> dict:
    """Is the library's best chemistry concentrated in a few clusters?

    The cheapest possible test of the whole idea, and it needs no optimiser.
    If the top ``top_frac`` of compounds is spread across clusters in
    proportion to cluster size, every region is equally good and there is
    nothing for a chooser to choose.

    ``lift`` is the ratio of the best cluster's share of the top compounds to
    its share of the library. Lift 1.0 means perfectly uniform -- no signal.
    ``coverage_best`` is the fraction of all top compounds sitting in that one
    cluster, which bounds what restricting to it could ever find.
    """
    labels, values = np.asarray(labels), np.asarray(values, dtype=float)
    n_top = max(1, int(round(top_frac * len(values))))
    top = set(np.argsort(-values)[:n_top].tolist())
    rows = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        hits = len(top.intersection(idx.tolist()))
        share_lib = len(idx) / len(values)
        rows.append({"cluster": int(c), "size": int(len(idx)), "n_top": hits,
                     "share_of_top": hits / n_top,
                     "lift": (hits / n_top) / share_lib if share_lib else 0.0,
                     "best_value": float(values[idx].max()) if len(idx) else float("nan")})
    rows.sort(key=lambda r: -r["n_top"])
    best = rows[0]
    # How many clusters must you hold to cover half the top compounds? A small
    # number means the signal is concentrated and worth chasing.
    cum, need_half = 0, 0
    for r in rows:
        cum += r["n_top"]
        need_half += 1
        if cum >= 0.5 * n_top:
            break
    return {"n_clusters": int(len(rows)), "n_top": n_top,
            "best_cluster": best["cluster"], "best_lift": best["lift"],
            "coverage_best": best["share_of_top"],
            "clusters_for_half_the_top": need_half,
            "per_cluster": rows}


def coherence(labels: np.ndarray, smiles: Sequence[str], sample: int = 1500,
              seed: int = 0) -> dict:
    """Are clusters chemically meaningful, or just partitions?

    Mean Tanimoto similarity within a cluster against between clusters. A
    coherent partition separates them clearly; k-means on fingerprints usually
    does not, because Euclidean distance is not Tanimoto.
    """
    from .surrogate import fingerprint_matrix, tanimoto_gram

    smiles = list(smiles)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(smiles), sample, replace=False)
           if len(smiles) > sample else np.arange(len(smiles)))
    X = fingerprint_matrix([smiles[i] for i in idx])
    K = tanimoto_gram(X, X)
    lab = np.asarray(labels)[idx]
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(idx), k=1)
    s, b = same[iu], K[iu]
    sizes = np.bincount(np.asarray(labels))
    return {"within": float(b[s].mean()) if s.any() else float("nan"),
            "between": float(b[~s].mean()) if (~s).any() else float("nan"),
            "separation": float(b[s].mean() - b[~s].mean()) if s.any() and (~s).any() else float("nan"),
            "size_min": int(sizes[sizes > 0].min()), "size_median": int(np.median(sizes[sizes > 0])),
            "size_max": int(sizes.max()), "n_singletons": int((sizes == 1).sum())}


# --------------------------------------------------------------------------
# constraint sources: choosing a region each round
# --------------------------------------------------------------------------


@dataclass
class ClusterConstraint(Constraint):
    """Allow only the members of ``clusters``."""

    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    clusters: tuple[int, ...] = ()
    kind: str = "cluster"

    def mask(self, ctx: LibraryContext) -> np.ndarray:
        return np.isin(self.labels, np.asarray(self.clusters, dtype=int))

    def describe(self) -> dict:
        """Must distinguish one region from another.

        ``LibraryContext.mask_for`` caches on ``fingerprint()``, which hashes
        this dict -- and the base implementation returns only ``{"kind": ...}``.
        Inheriting it silently gave every region the first one's mask, so a
        bandit, a random region and the hindsight-best region all scored
        identically. The labels are hashed too: cluster 3 under k=10 is not
        cluster 3 under k=30.
        """
        import hashlib

        digest = hashlib.sha256(np.asarray(self.labels).tobytes()).hexdigest()[:12]
        return {"kind": self.kind, "clusters": sorted(int(c) for c in self.clusters),
                "labels": digest}


@dataclass
class FixedClusterSource:
    """Always the same region. Used for the two reference lines.

    With the hindsight-best cluster this is the **ceiling** for any chooser
    that commits to one region; with a random cluster it is the floor. Neither
    is a method -- the first reads the answer key.
    """

    labels: np.ndarray
    cluster: int

    def __call__(self, **_) -> ClusterConstraint:
        return ClusterConstraint(labels=self.labels, clusters=(int(self.cluster),))


@dataclass
class HeuristicClusterSource:
    """Pick one region by a fixed rule, with no model and no measurements.

    ``largest`` is the one that matters. On 121 held-out BindingDB targets it
    identified the region holding the most top-1% compounds 24.8% of the time
    against a 10% random floor, beating a learned protein-conditioned prior --
    so it has to be run as a campaign arm before any model is wired in. It is
    also a *weak* restriction by construction, which cuts both ways: a big
    region starves BO less, but it also focuses the search less.
    """

    labels: np.ndarray
    centroids: np.ndarray | None = None
    rule: str = "largest"
    values: np.ndarray | None = None
    """Only for ``densest``, which reads the answer key like the hindsight
    ceiling does. It is a second reference line, not a method: it asks whether
    the right region to aim for is the one with the *most* good compounds or
    the one with the highest *concentration* of them -- which decides what a
    learned chooser should be trained to predict."""
    top_frac: float = 0.01
    _pick: int = field(default=-1, init=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        sizes = np.bincount(self.labels, minlength=int(self.labels.max()) + 1).astype(float)
        if self.rule == "largest":
            self._pick = int(np.argmax(sizes))
        elif self.rule == "densest":
            if self.values is None:
                raise ValueError("rule 'densest' needs values")
            v = np.asarray(self.values, dtype=float)
            n_top = max(1, int(round(self.top_frac * len(v))))
            top = set(np.argsort(-v)[:n_top].tolist())
            dens = np.array([
                len(top.intersection(np.flatnonzero(self.labels == c).tolist())) /
                max(1.0, sizes[c]) for c in range(len(sizes))])
            self._pick = int(np.argmax(dens))
        elif self.rule in ("dense", "diverse"):
            if self.centroids is None:
                raise ValueError(f"rule {self.rule!r} needs centroids")
            c = np.asarray(self.centroids, dtype=float)
            c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
            sim = c @ c.T
            np.fill_diagonal(sim, np.nan)
            mean_sim = np.nanmean(sim, axis=1)
            self._pick = int(np.argmax(mean_sim if self.rule == "dense" else -mean_sim))
        else:
            raise ValueError(f"unknown rule {self.rule!r}")

    def __call__(self, **_) -> "ClusterConstraint":
        return ClusterConstraint(labels=self.labels, clusters=(self._pick,))


@dataclass
class UCBClusterSource:
    """A bandit over regions: the baseline a learned chooser must beat.

    Each cluster's value is the best objective seen inside it so far; the bonus
    is the usual ``c * sqrt(log t / n_c)``, so a region left alone becomes
    attractive again. Unvisited regions are pulled first, which is exactly
    where this breaks down on a shattered library: with 299 clusters and 200
    calls the bandit spends its whole budget on first visits and never
    exploits. Pass a coarse ``k`` or expect it to reduce to random.

    ``prior`` is where a learned chooser plugs in -- an initial value per
    cluster, before any sampling. That is the only thing a model can offer
    that this baseline lacks, and its influence decays as ``n_c`` grows.
    """

    labels: np.ndarray
    c: float = 1.0
    prior: np.ndarray | None = None
    prior_weight: float = 1.0
    """Pseudo-counts given to ``prior``. Higher means the prior survives more
    evidence before the data overrides it."""

    n_clusters: int = field(init=False)
    log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        self.n_clusters = int(self.labels.max()) + 1

    def reset(self) -> None:
        self.log = []

    def __call__(self, observations=None, ctx=None, rng=None, round_index: int = 0,
                 seed: int = 0, **_) -> ClusterConstraint:
        obs = list(observations or [])
        best = np.full(self.n_clusters, -np.inf)
        count = np.zeros(self.n_clusters)
        if obs and ctx is not None:
            pos = {s: i for i, s in enumerate(ctx.smiles)}
            for smi, val in obs:
                i = pos.get(smi)
                if i is None:
                    continue
                c = int(self.labels[i])
                best[c] = max(best[c], float(val))
                count[c] += 1

        if self.prior is not None:
            p = np.asarray(self.prior, dtype=float)
            seen = np.isfinite(best) & (count > 0)
            # Blend the prior with observed value, weighting by evidence.
            blended = np.where(seen,
                               (self.prior_weight * p + count * np.where(seen, best, 0.0))
                               / (self.prior_weight + count),
                               p)
        else:
            blended = np.where(count > 0, best, np.nan)

        t = max(1.0, count.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.c * np.sqrt(np.log(t + 1.0) / np.maximum(count, 1e-9))
        score = np.where(count > 0, np.nan_to_num(blended, nan=0.0) + bonus, np.inf)
        if self.prior is not None:
            # With a prior, an unvisited cluster is not infinitely attractive --
            # the prior already says something about it. This is the whole point
            # of having one on a library with more clusters than budget.
            score = np.nan_to_num(blended, nan=0.0) + bonus
        pick = int(np.argmax(score))
        self.log.append({"round": int(round_index), "cluster": pick,
                         "n_seen": int(count[pick]), "visited": int((count > 0).sum())})
        return ClusterConstraint(labels=self.labels, clusters=(pick,))

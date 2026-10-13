"""A learned prior over regions of chemical space, conditioned on the target.

The one thing a model can offer that a bandit over regions cannot: an opinion
about which region holds the actives **before any compound is measured**. A
bandit must spend samples to find that out, and at a small budget it never
finishes paying -- measured on cruzipain at 10 regions and 30 oracle calls, UCB
captured 2% of the available headroom because it spent the whole budget on
first visits.

Shape, which is why a contrastive scorer rather than a generative model:

    protein sequence  -> ESM2 (frozen)      -> protein head  --.
                                                                >-- cosine -> one score per region
    region centroid   -> mean ECFP4          -> region head  --'

The action space is the k regions, scored in a single forward pass. Nothing is
generated, so there is no sampling, no token budget, and no parse to fail; and
scoring 500 regions costs the same as scoring 5, which is the property an
autoregressive agent cannot match.

Both encoders are frozen and classical on purpose. Fingerprints beat every
neural molecular encoder measured on this data (Spearman 0.64-0.74 against
0.20-0.50 for ChemBERTa, MoLFormer and Qwen3.5-4B), and ESM2 is the only side
where a language model is the right instrument at all -- it reads amino acids,
not SMILES.

The controls that decide whether this learned anything are in
``scripts/train_prior.py``: pick the largest region, pick the most diverse
region, and -- the one that matters -- shuffle the protein embeddings between
targets. If shuffling does not hurt, the protein contributed nothing and the
model is just ranking regions by shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def make_head(in_dim: int, width: int = 256, proj: int = 128, depth: int = 2,
              dropout: float = 0.1):
    """``in_dim -> width -> proj`` MLP.

    Deliberately small. The v1 selector used 14.2M parameters against 7,600
    correlated examples and overfit by epoch 2; here there are a few hundred
    targets, so the head is sized in the tens of thousands.
    """
    import torch.nn as nn

    layers: list = [nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
    layers += [nn.Linear(width, proj)]
    return nn.Sequential(*layers)


class ClusterPrior:
    """Scores each region of a library for a given protein, with no measurements."""

    def __init__(self, protein_dim: int, region_dim: int, width: int = 256,
                 proj: int = 128, depth: int = 2, dropout: float = 0.1,
                 device: str | None = None):
        import torch
        import torch.nn as nn

        self.cfg = dict(protein_dim=protein_dim, region_dim=region_dim, width=width,
                        proj=proj, depth=depth, dropout=dropout)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.protein_head = make_head(protein_dim, width, proj, depth, dropout).to(self.device)
        self.region_head = make_head(region_dim, width, proj, depth, dropout).to(self.device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=self.device))

    def parameters(self):
        return [*self.protein_head.parameters(), *self.region_head.parameters(),
                self.logit_scale]

    def train(self, mode: bool = True) -> "ClusterPrior":
        self.protein_head.train(mode)
        self.region_head.train(mode)
        return self

    def logits(self, protein, regions):
        """``protein [b, dp]``, ``regions [b, k, dr]`` -> ``[b, k]``."""
        import torch
        import torch.nn.functional as F

        zp = F.normalize(self.protein_head(protein), dim=-1)
        b, k, d = regions.shape
        zr = F.normalize(self.region_head(regions.reshape(b * k, d)), dim=-1).view(b, k, -1)
        return self.logit_scale.exp().clamp(max=100.0) * torch.einsum("bh,bkh->bk", zp, zr)

    def score(self, protein_emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """-> one score per region, higher is more promising. Inference only."""
        import torch

        self.train(False)
        with torch.no_grad():
            p = torch.as_tensor(protein_emb, dtype=torch.float32, device=self.device).reshape(1, -1)
            r = torch.as_tensor(centroids, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.logits(p, r)[0].cpu().numpy()

    def save(self, path: str | Path, **meta) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protein_head": self.protein_head.state_dict(),
                    "region_head": self.region_head.state_dict(),
                    "logit_scale": self.logit_scale.detach().cpu(),
                    "cfg": self.cfg, "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "ClusterPrior":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(device=device, **ck["cfg"])
        m.protein_head.load_state_dict(ck["protein_head"])
        m.region_head.load_state_dict(ck["region_head"])
        with torch.no_grad():
            m.logit_scale.copy_(ck["logit_scale"].to(m.device))
        m.meta = ck.get("meta", {})
        return m.train(False)


# --------------------------------------------------------------------------
# free baselines the prior has to beat
# --------------------------------------------------------------------------


def baseline_scores(centroids: np.ndarray, sizes: np.ndarray, kind: str,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Region scores from rules that need no model and no target.

    ``largest``
        Pick the biggest region. Strong and annoying: a big region contains
        more compounds, so it contains more good ones by chance alone, and it
        gives BO more room before the mask starves.
    ``diverse``
        Pick the region whose centroid is least like the others -- the
        odd-one-out. Cheap proxy for "unusual chemistry".
    ``dense``
        Pick the region whose centroid is most like the others, i.e. the
        mainstream chemotype of the library.
    ``random``
        The floor: 1/k accuracy.
    """
    k = len(centroids)
    if kind == "largest":
        return sizes.astype(float)
    if kind == "random":
        return (rng or np.random.default_rng(0)).random(k)
    c = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    sim = c @ c.T
    np.fill_diagonal(sim, np.nan)
    mean_sim = np.nanmean(sim, axis=1)
    if kind == "diverse":
        return -mean_sim
    if kind == "dense":
        return mean_sim
    raise ValueError(f"unknown baseline {kind!r}")


@dataclass
class PriorClusterSource:
    """Feeds a trained prior into the campaign as a warm bandit.

    Wraps :class:`molbo.clusters.UCBClusterSource`, supplying its ``prior``
    argument. The bandit still updates on evidence, so the prior only decides
    the early rounds -- which is exactly the regime it is meant to help, and
    means a bad prior is recoverable rather than absorbing.
    """

    labels: np.ndarray
    scores: np.ndarray
    c: float = 1.0
    prior_weight: float = 2.0

    _inner: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from .clusters import UCBClusterSource

        s = np.asarray(self.scores, dtype=float)
        # Put the prior on the objective's scale: the bandit compares it with
        # observed values, so raw cosine logits would be either ignored or
        # overwhelming depending on the temperature.
        s = (s - s.mean()) / (s.std() + 1e-9)
        self._inner = UCBClusterSource(labels=self.labels, c=self.c, prior=s,
                                       prior_weight=self.prior_weight)

    def reset(self) -> None:
        self._inner.reset()

    def __call__(self, **kw):
        return self._inner(**kw)

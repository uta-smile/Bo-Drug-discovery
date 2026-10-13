"""A contrastive (CLM-style) selector: frozen LM encoder, two trained heads.

This is the selection arm (:class:`molbo.llm.LLMSelector`) with the model's
*generation* taken out. BO proposes a shortlist exactly as before; instead of
asking a chat model to name an index, each candidate is scored by how well its
embedding aligns with an embedding of the SAR observed so far, and the argmax
is bought.

The recipe is Contrastive Language Models (Kwok et al. 2026,
github.com/Contrastive-LM/CLM, Apache-2.0): a frozen decoder LM, last-token
pooling, and two small MLP heads -- one for the state, one for the candidate --
trained so that ``exp(logit_scale) * cos(state_head(s), action_head(c))`` ranks
candidates. :func:`make_head` is a port of their ``clm.heads.make_head``. What
changes here is the backbone and the data:

* The backbone is ``Qwen/Qwen3.5-4B-Base`` (2560-d), the same weights family the
  llama.cpp arms run. The released CLM-8B heads were fitted to Qwen3-8B's
  4096-d space and do not transfer, so heads are trained from scratch on
  shortlists recorded from plain-BO runs on *other* ChEMBL targets
  (``scripts/clm_collect.py``, ``scripts/clm_train.py``). EGFR is never in the
  training pool, so the lookup tier's EGFR series stay a clean test.

Why this shape fits the project better than a chat selector:

* **No positional bias by construction.** Each candidate is embedded alone and
  scored independently of where it sits in the list, so the primacy/recency
  artefact measured on Qwen3.5-4B (see ``LLMSelector.shuffle``) cannot occur.
* **Deterministic and parse-free.** A forward pass, not sampling; there is no
  JSON to malform and no answer to truncate.
* **Candidate embeddings are reusable.** A SMILES is embedded once per library,
  ever; each round re-embeds only the state text.

What it gives up: state and candidate never attend to each other, so any
interaction the model could use must survive being squeezed into two vectors.
The controls that decide whether that matters are the same as for the chat
selector -- :class:`~molbo.llm.RandomShortlistSelector`, and simply buying the
shortlist's highest posterior mean.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .llm import format_observations

BACKBONE = "Qwen/Qwen3.5-4B-Base"
BACKBONE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
"""Pinned: an embedding cache is only valid for the exact weights that made it."""

UNITS = "pIC50 (higher is more potent)"
MAX_SHOW = 24


def state_text(observations: Sequence[tuple[str, float]], units: str = UNITS,
               max_show: int = MAX_SHOW) -> str:
    """The state the state head reads: the same SAR table the chat selector sees."""
    return format_observations(observations, max_show=max_show, units=units)


# ---- encoder ---------------------------------------------------------------


@dataclass
class Encoder:
    """Frozen backbone, last-token pooling, L2-normalised, cached on disk.

    The cache is one ``.npz`` per backbone key holding ``keys`` (sha1 of each
    text) and ``vecs`` (float16). Texts are embedded in a canonical order --
    sorted by length, then text -- so a given text always shares a batch with
    the same neighbours and the cache is reproducible bit-for-bit. Batch
    composition was checked to move an embedding by cos 0.9999 (bf16
    rounding), so this is belt-and-braces rather than a correction.
    """

    model: str = BACKBONE
    revision: str | None = BACKBONE_REVISION
    pooling: str = "last"
    """``last`` (final token) or ``mean`` (mask-weighted average over tokens).

    Not cosmetic. Last-token pooling is the right read for a causal model whose
    final position has attended to everything, but on a *base* model it is a
    known-weak sentence embedding, and the v1 probe measured exactly that:
    Spearman 0.21-0.27 against a fingerprint's 0.64-0.73. Encoder models
    (ChemBERTa, MoLFormer) are bidirectional and want ``mean``.
    """
    trust_remote_code: bool = False
    """Required by MoLFormer, which ships a custom architecture."""
    cache_dir: str | Path = "data/cache/clm"
    max_len: int = 2048
    token_budget: int = 4096
    """Padded tokens per forward pass; batch size adapts to text length. With the
    flash-linear-attention kernels, 4096 ran ~1.5x faster than 16384 on a
    16 GB card (0.29 vs 0.43 s per ~1100-token state)."""
    checkpoint_every: int = 500
    """Texts between cache writes. At ~0.3 s per state that is a couple of
    minutes of work at risk, which is the right trade against rewriting a
    growing ``.npz``."""
    device: str | None = None

    _tok: object = field(default=None, init=False, repr=False)
    _lm: object = field(default=None, init=False, repr=False)
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _vecs: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _dirty: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        p = self.cache_path
        if p.exists():
            z = np.load(p)
            self._index = {k: i for i, k in enumerate(z["keys"].tolist())}
            self._vecs = list(z["vecs"])

    def key(self) -> str:
        return f"{self.model}@{(self.revision or 'main')[:12]}:{self.pooling}:{self.max_len}"

    @property
    def cache_path(self) -> Path:
        slug = hashlib.sha1(self.key().encode()).hexdigest()[:12]
        return self.cache_dir / f"emb_{slug}.npz"

    @staticmethod
    def text_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self._lm is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"revision": self.revision, "trust_remote_code": self.trust_remote_code}
        self._tok = AutoTokenizer.from_pretrained(self.model, **kw)
        # Left padding puts the real final token at index -1, which is what
        # ``last`` reads. ``mean`` uses the attention mask, so either side works.
        self._tok.padding_side = "left" if self.pooling == "last" else "right"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token or self._tok.sep_token
        # float32 for the small encoders: bf16 buys nothing under ~100M params
        # and costs precision the probe is trying to measure.
        dtype = torch.bfloat16 if self._is_large() else torch.float32
        self._lm = AutoModel.from_pretrained(self.model, dtype=dtype, device_map=dev, **kw).eval()
        self.device = dev

    def _is_large(self) -> bool:
        return any(t in self.model for t in ("4B", "7B", "8B", "650M", "3B"))

    def _forward(self, texts: list[str]) -> np.ndarray:
        import torch

        batch = self._tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=self.max_len).to(self.device)
        with torch.no_grad():
            out = self._lm(**batch).last_hidden_state
        if self.pooling == "mean":
            m = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            h = (out * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "last":
            h = out[:, -1]
        else:
            raise ValueError(f"unknown pooling {self.pooling!r}; expected 'last' or 'mean'")
        return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()

    def _forward_resilient(self, texts: list[str]) -> np.ndarray:
        """``_forward``, halving the batch on a CUDA fault before giving up.

        A long pass of 1000-token states on a consumer Windows card died once
        with ``CUDA error: unknown error`` after ~35 minutes at full occupancy
        -- the signature of a driver timeout (TDR) on a long kernel, not of a
        bug in the batch. One text at a time is the smallest kernel available,
        so a halving retry converts that class of fault into lost seconds
        rather than a lost run. Combined with :meth:`flush` checkpointing, a
        genuinely fatal fault costs only the work since the last checkpoint.
        """
        import torch

        try:
            return self._forward(texts)
        except (torch.AcceleratorError, RuntimeError):
            if len(texts) == 1:
                raise
            self.flush()
            mid = len(texts) // 2
            return np.concatenate([self._forward_resilient(texts[:mid]),
                                   self._forward_resilient(texts[mid:])])

    def embed(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """-> ``[n, hidden]`` float32, L2-normalised. Misses are computed and cached."""
        keys = [self.text_key(t) for t in texts]
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in self._index}.items(),
                      key=lambda kv: (len(kv[1]), kv[1]))
        if todo:
            self._load()
            lengths = [len(self._tok(t, add_special_tokens=False)["input_ids"]) for _, t in todo]
            i, done = 0, 0
            while i < len(todo):
                # Sorted ascending, so the batch's longest text is its last one.
                j = i + 1
                while j < len(todo) and (j - i + 1) * min(lengths[j], self.max_len) <= self.token_budget:
                    j += 1
                chunk = todo[i:j]
                for (k, _), v in zip(chunk, self._forward_resilient([t for _, t in chunk])):
                    self._index[k] = len(self._vecs)
                    self._vecs.append(v.astype(np.float16))
                done += len(chunk)
                i = j
                if done // self.checkpoint_every != (done - len(chunk)) // self.checkpoint_every:
                    # Checkpoint: a crash an hour into a long pass keeps its work.
                    self._dirty = True
                    self.flush()
                if progress:
                    print(f"\r[clm] embedded {done}/{len(todo)}", end="", flush=True)
            if progress:
                print()
            self._dirty = True
            self.flush()
        return np.stack([self._vecs[self._index[k]] for k in keys]).astype(np.float32)

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(self._index, key=self._index.get))
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vecs=np.stack(self._vecs))
        os.replace(tmp, self.cache_path)
        self._dirty = False

    @property
    def hidden_size(self) -> int | None:
        return int(self._vecs[0].shape[0]) if self._vecs else None


# ---- heads -----------------------------------------------------------------


def make_head(hidden: int, width: int = 1536, depth: int = 3, proj: int = 512,
              activation: str = "gelu", layernorm: bool = True, residual: bool = False):
    """``hidden -> width -> ... -> proj`` MLP. Port of CLM's ``clm.heads.make_head``."""
    import torch.nn as nn

    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class Heads:
    """State head + action head + temperature, as one checkpoint."""

    def __init__(self, cfg: dict, device: str | None = None):
        import torch
        import torch.nn as nn

        self.cfg = dict(cfg)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {k: cfg[k] for k in ("hidden", "width", "depth", "proj", "activation",
                                  "layernorm", "residual")}
        self.state_head = make_head(**kw).to(self.device)
        self.action_head = make_head(**kw).to(self.device)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=self.device))

    def parameters(self):
        return [*self.state_head.parameters(), *self.action_head.parameters(), self.logit_scale]

    def train(self, mode: bool = True) -> "Heads":
        self.state_head.train(mode)
        self.action_head.train(mode)
        return self

    def logits(self, states, cands):
        """``states [b, h]``, ``cands [b, k, h]`` -> ``[b, k]`` scaled cosine scores."""
        import torch
        import torch.nn.functional as F

        zs = F.normalize(self.state_head(states), dim=-1)
        b, k, h = cands.shape
        zc = F.normalize(self.action_head(cands.reshape(b * k, h)), dim=-1).view(b, k, -1)
        return self.logit_scale.exp().clamp(max=100.0) * torch.einsum("bh,bkh->bk", zs, zc)

    def save(self, path: str | Path, **meta) -> None:
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_head": self.state_head.state_dict(),
                    "action_head": self.action_head.state_dict(),
                    "logit_scale": self.logit_scale.detach().cpu(),
                    "cfg": self.cfg, "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Heads":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        heads = cls(ck["cfg"], device=device)
        heads.state_head.load_state_dict(ck["state_head"])
        heads.action_head.load_state_dict(ck["action_head"])
        with torch.no_grad():
            heads.logit_scale.copy_(ck["logit_scale"].to(heads.device))
        heads.meta = ck.get("meta", {})
        return heads.train(False)


# ---- selectors ---------------------------------------------------------------


@dataclass
class CLMSelector:
    """BO proposes a shortlist; the trained heads pick the argmax.

    Same call signature and bookkeeping fields as :class:`molbo.llm.LLMSelector`
    so ``run_lookup_experiment.py`` reports it identically. ``failures`` and
    ``deferred`` stay zero -- there is nothing to parse -- and exist only so
    the two arms read the same in a table.
    """

    heads: Heads
    encoder: Encoder
    units: str = UNITS
    max_show: int = MAX_SHOW

    calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    deferred: int = field(default=0, init=False)
    ranks: list[int] = field(default_factory=list, init=False)
    probs: list[list[float]] = field(default_factory=list, init=False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, encoder: Encoder | None = None, **kw) -> "CLMSelector":
        heads = Heads.load(path)
        enc = encoder or Encoder()
        want = heads.cfg.get("encoder")
        if want and want != enc.key():
            raise ValueError(f"heads were trained on {want!r}, encoder is {enc.key()!r}")
        return cls(heads=heads, encoder=enc, **kw)

    def score(self, shortlist: Sequence[str], observations) -> np.ndarray:
        import torch

        s = self.encoder.embed([state_text(observations, self.units, self.max_show)])
        c = self.encoder.embed(list(shortlist))
        dev = self.heads.device
        with torch.no_grad():
            lg = self.heads.logits(torch.from_numpy(s).to(dev), torch.from_numpy(c)[None].to(dev))
        return lg[0].cpu().numpy()

    def __call__(self, shortlist, observations, **_):
        lg = self.score(shortlist, observations)
        self.calls += 1
        p = np.exp(lg - lg.max())
        self.probs.append((p / p.sum()).round(4).tolist())
        rank = int(np.argmax(lg))
        self.ranks.append(rank)
        return rank


@dataclass
class MaxMeanSelector:
    """Control: buy the shortlist's highest posterior mean.

    Free, and the thing a learnt selector is most likely to rediscover -- on
    recorded EI shortlists it beats EI's own pick at choosing the most potent
    candidate. A CLM arm that cannot beat this has learnt nothing the GP did
    not already know.
    """

    def __call__(self, mu, **_):
        return int(np.argmax(np.asarray(mu)))


@dataclass
class OracleShortlistSelector:
    """Ceiling, not a method: always buy the shortlist's truly best compound.

    Reads the answer key, so it bounds what *any* selector on this hook could
    achieve. Run it before training one, for the same reason
    ``scripts/acquisition_headroom.py`` runs ``OraclePolicy`` before any
    controller: if the ceiling cannot beat plain BO, no selector can, and the
    honest conclusion is that the shortlist hook is the wrong place to
    intervene rather than that a given selector failed.

    Worth stating why that is not a foregone conclusion. Perfect *greedy*
    selection is not perfect selection: it buys the most potent candidate every
    round and therefore never spends a call resolving uncertainty, which is
    exactly what :class:`MaxMeanSelector` was measured doing -- +1.053 offline
    gain, and campaign AUC -0.0195 (p=0.011) with top-1% recall halved against
    plain BO. The ceiling says whether the hook has headroom at all.
    """

    values: dict[str, float]

    def __call__(self, shortlist, **_):
        return int(np.argmax([self.values[s] for s in shortlist]))


@dataclass
class RecordingSelector:
    """Always returns rank 0 -- so the run *is* plain BO -- and records the state.

    Used to harvest training data that matches the test distribution exactly:
    the shortlists the selector will face are EI top-k lists from a GP fitted
    to BO's own trajectory, so that is what it is trained on. ``values`` is the
    lookup table, used only to label the shortlist after the fact; nothing it
    holds reaches the campaign.
    """

    values: dict[str, float]
    tag: str = ""
    records: list[dict] = field(default_factory=list, init=False)

    def __call__(self, shortlist, scores, mu, sd, observations, round_index: int = 0,
                 seed: int = 0, **_):
        self.records.append({
            "dataset": self.tag, "seed": int(seed), "round": int(round_index),
            "observations": [[s, float(v)] for s, v in observations],
            "shortlist": list(shortlist),
            "true": [float(self.values[s]) for s in shortlist],
            "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
            "acq": [float(x) for x in scores],
        })
        return 0


# ---- offline evaluation ------------------------------------------------------


def shortlist_metrics(scores: Sequence[np.ndarray], true: Sequence[np.ndarray]) -> dict:
    """How good is buying ``argmax(scores)`` from each shortlist?

    ``gain``: true value of the pick minus the shortlist mean (0 = random).
    ``regret``: shortlist max minus the pick (0 = oracle).
    ``top1``: fraction of shortlists where the pick is the true best.
    """
    gain, regret, top1 = [], [], []
    for s, t in zip(scores, true):
        t = np.asarray(t, dtype=float)
        i = int(np.argmax(s))
        gain.append(t[i] - t.mean())
        regret.append(t.max() - t[i])
        top1.append(float(t[i] >= t.max()))
    n = max(1, len(gain))
    return {"n": len(gain), "gain": float(np.mean(gain)) if gain else float("nan"),
            "gain_se": float(np.std(gain) / math.sqrt(n)) if gain else float("nan"),
            "regret": float(np.mean(regret)) if regret else float("nan"),
            "top1": float(np.mean(top1)) if top1 else float("nan")}

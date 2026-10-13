"""Train CLM heads on recorded plain-BO shortlists; evaluate offline against controls.

    python scripts/clm_train.py --train data/clm/train.jsonl \
        --test data/clm/egfr_test.jsonl --val-dataset cdk2_all \
        --out data/clm/heads_v1.pt

    python scripts/clm_train.py --train data/clm/train.jsonl \
        --test data/clm/egfr_test.jsonl --embed-only      # warm the cache, no training

Each record is one shortlist from ``scripts/clm_collect.py``. The state is the
observed SAR (the text the chat selector reads), the candidates are the
shortlist's SMILES, and the target is ``softmax(true / tau)`` over the shortlist
-- mass on whichever candidates are actually most potent. Loss is the soft
cross-entropy of CLM's ``choice`` task with ``--loss softce``: a softmax over
each shortlist's own candidates. In-batch negatives would be molecules from
other targets, which are trivially separable and teach nothing.

Validation holds out a whole *target* (``--val-dataset``), not random rounds,
because rounds of one run share most of their observations and would leak. The
test file must be a target that never appears in training; the default story
is five kinase/esterase targets in, EGFR out.

Every split is scored against the controls that decide whether the heads learnt
anything: ``random`` (gain 0 by definition), ``bo_rank0`` (plain BO's own pick),
and ``max_mu`` (buy the highest posterior mean -- free, and the thing a learnt
selector most plausibly rediscovers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.clm import Encoder, Heads, shortlist_metrics, state_text  # noqa: E402


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_records(enc: Encoder, recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> states [n, h], candidate index [n, k], candidate matrix [m, h], true [n, k]."""
    states = [state_text([tuple(o) for o in r["observations"]]) for r in recs]
    smiles = sorted({s for r in recs for s in r["shortlist"]})
    col = {s: i for i, s in enumerate(smiles)}
    print(f"[clm] {len(recs)} states, {len(smiles)} unique candidates", flush=True)
    C = enc.embed(smiles, progress=True)
    S = enc.embed(states, progress=True)
    k = max(len(r["shortlist"]) for r in recs)
    idx = np.array([[col[s] for s in r["shortlist"]] + [-1] * (k - len(r["shortlist"])) for r in recs])
    true = np.array([r["true"] + [np.nan] * (k - len(r["true"])) for r in recs], dtype=np.float32)
    return S, idx, C, true


def baselines(recs: list[dict]) -> dict:
    t = [np.asarray(r["true"]) for r in recs]
    return {
        "bo_rank0": shortlist_metrics([-np.arange(len(r["true"])) for r in recs], t),
        "max_mu": shortlist_metrics([np.asarray(r["mu"]) for r in recs], t),
        "oracle": shortlist_metrics(t, t),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", nargs="*", default=[])
    p.add_argument("--val-dataset", default="cdk2_all", help="training-pool target held out for early stopping")
    p.add_argument("--out", default="data/clm/heads.pt")
    p.add_argument("--embed-only", action="store_true")
    p.add_argument("--tau", type=float, default=0.25, help="target temperature, log units")
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--proj", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    enc = Encoder()
    train_all = load(args.train)
    tests = {Path(t).stem: load(t) for t in args.test}
    leaked = {r["dataset"] for r in train_all} & {r["dataset"] for v in tests.values() for r in v}
    if leaked:
        print(f"error: test dataset(s) {sorted(leaked)} also appear in training", file=sys.stderr)
        return 2

    t0 = time.time()
    data = {"train_pool": embed_records(enc, train_all)}
    for name, recs in tests.items():
        data[name] = embed_records(enc, recs)
    print(f"[clm] embeddings ready in {time.time() - t0:.0f}s ({enc.key()})", flush=True)
    if args.embed_only:
        return 0

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def tensors(S, idx, C, true, rows=None):
        rows = np.arange(len(S)) if rows is None else rows
        return (torch.from_numpy(S[rows]).to(dev), torch.from_numpy(idx[rows]).to(dev),
                torch.from_numpy(C).to(dev), torch.from_numpy(true[rows]).to(dev))

    S, idx, C, true = data["train_pool"]
    is_val = np.array([r["dataset"] == args.val_dataset for r in train_all])
    if not is_val.any():
        print(f"error: --val-dataset {args.val_dataset!r} not in training file", file=sys.stderr)
        return 2
    splits = {"train": (tensors(S, idx, C, true, np.where(~is_val)[0]),
                        [r for r, v in zip(train_all, is_val) if not v]),
              "val": (tensors(S, idx, C, true, np.where(is_val)[0]),
                      [r for r, v in zip(train_all, is_val) if v])}
    for name, recs in tests.items():
        splits[name] = (tensors(*data[name]), recs)

    cfg = dict(hidden=int(S.shape[1]), width=args.width, depth=args.depth, proj=args.proj,
               activation="gelu", layernorm=True, residual=False, encoder=enc.key(), tau=args.tau)
    heads = Heads(cfg, device=dev)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"[clm] heads {n_params / 1e6:.1f}M params | train {len(splits['train'][1])} "
          f"| val ({args.val_dataset}) {len(splits['val'][1])}", flush=True)

    def logits_of(St, It, Ct):
        mask = It < 0
        lg = heads.logits(St, Ct[It.clamp(min=0)])
        return lg.masked_fill(mask, float("-inf"))

    def soft_target(Tt):
        return torch.softmax((Tt / args.tau).nan_to_num(nan=-1e4), dim=-1)

    @torch.no_grad()
    def evaluate(split):
        (St, It, Ct, Tt), recs = splits[split]
        heads.train(False)
        lg = torch.cat([logits_of(St[i:i + 1024], It[i:i + 1024], Ct) for i in range(0, len(St), 1024)])
        loss = -(soft_target(Tt) * F.log_softmax(lg, -1).nan_to_num(neginf=0.0)).sum(-1).mean().item()
        heads.train(True)
        scores = [row[: len(r["true"])] for row, r in zip(lg.cpu().numpy(), recs)]
        m = shortlist_metrics(scores, [np.asarray(r["true"]) for r in recs])
        m["loss"] = loss
        return m

    (St, It, Ct, Tt), _ = splits["train"]
    best, best_state, bad = -math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(St), device=dev)
        tot = 0.0
        for i in range(0, len(St), args.batch):
            b = perm[i:i + args.batch]
            lg = logits_of(St[b], It[b], Ct)
            loss = -(soft_target(Tt[b]) * F.log_softmax(lg, -1).nan_to_num(neginf=0.0)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        v = evaluate("val")
        print(f"[clm] epoch {epoch:2d}  train loss {tot / len(St):.4f}  val loss {v['loss']:.4f}  "
              f"val gain {v['gain']:+.3f}  top1 {v['top1']:.3f}", flush=True)
        if v["gain"] > best:
            best, bad = v["gain"], 0
            best_state = {k: {n: t.detach().clone() for n, t in m.state_dict().items()}
                          for k, m in (("s", heads.state_head), ("a", heads.action_head))}
            best_scale = heads.logit_scale.detach().clone()
            best_epoch = epoch
        else:
            bad += 1
            if bad >= args.patience:
                break

    heads.state_head.load_state_dict(best_state["s"])
    heads.action_head.load_state_dict(best_state["a"])
    with torch.no_grad():
        heads.logit_scale.copy_(best_scale)

    report = {}
    print(f"\n[clm] best epoch {best_epoch}. gain = pick minus shortlist mean (pIC50); random = 0")
    print(f"{'split':<28s}{'selector':<10s}{'gain':>14s}{'regret':>9s}{'top1':>7s}")
    for split, (_, recs) in splits.items():
        rows = {"clm": evaluate(split), **baselines(recs)}
        report[split] = rows
        for name, m in rows.items():
            print(f"{split:<28s}{name:<10s}{m['gain']:+8.3f} +/-{m['gain_se']:.3f}"
                  f"{m['regret']:9.3f}{m['top1']:7.3f}")

    blob = hashlib.sha256(Path(args.train).read_bytes()).hexdigest()[:16]
    heads.save(args.out, train=args.train, train_sha256=blob, val_dataset=args.val_dataset,
               train_datasets=sorted({r["dataset"] for r in train_all}), best_epoch=best_epoch,
               args=vars(args), report=report)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"cfg": cfg, "train_sha256": blob, "best_epoch": best_epoch, "report": report}, indent=2))
    print(f"\n[clm] saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

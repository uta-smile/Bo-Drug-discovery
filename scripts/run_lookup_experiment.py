"""Run the primary-tier sweep: every arm, every seed, against measured pIC50.

    python scripts/run_lookup_experiment.py \
        --dataset data/chembl/egfr_anilinoquinazoline.csv \
        --budget 200 --seeds 20 --out results/egfr_primary

The oracle is a dictionary lookup, so the cost here is the GP, not the assay.
That is the entire point of the tier: 20 seeds x 4 arms is minutes, which is
what makes it possible to iterate on the method at all.

Reported per arm: AUC of the top-10 mean against unique oracle calls (the PMO
metric, computed exactly and also in PMO's own sampled form), the best pIC50
found, simple regret against the library's true optimum, and recall of the
library's true top 1%.  Every non-reference arm is tested against plain BO with
a paired signed-rank test over matched seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.campaign import ArmConfig  # noqa: E402
from molbo.experiment import (  # noqa: E402
    default_arms,
    run_sweep,
    save_sweep,
    summarize_sweep,
)
from molbo.oracles import LookupOracle  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="CSV built by build_chembl_series.py")
    p.add_argument("--budget", type=int, default=200, help="unique oracle calls per run")
    p.add_argument("--seeds", type=int, default=20, help="number of seeds per arm")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--n-init", type=int, default=10, help="random molecules before the GP starts")
    p.add_argument("--k", type=int, default=10, help="top-k for the AUC metric")
    p.add_argument(
        "--constraint-fraction",
        type=float,
        default=0.10,
        help="library fraction kept by the control and ceiling constraint arms",
    )
    p.add_argument("--arms", nargs="*", default=None, help="subset of arm names to run")
    p.add_argument(
        "--llm-arm",
        action="store_true",
        help="add bo_llm_constraint (needs anthropic credentials, or a warm cache)",
    )
    p.add_argument(
        "--llm-offline",
        action="store_true",
        help="refuse to call the API; replay the response cache only. Use this to "
        "prove a results table was reproduced purely from the record.",
    )
    p.add_argument("--llm-cache", default="data/cache/llm")
    p.add_argument(
        "--select-arm",
        action="store_true",
        help="add bo_llm_select (BO shortlists, the model picks) and its "
        "random-shortlist control. One model call per evaluation, so this is "
        "~40x more expensive than the constraint arm.",
    )
    p.add_argument(
        "--warmstart-arm",
        action="store_true",
        help="add bo_llm_warmstart (the model picks the initial design from a "
        "random sample) and its matched random control. One call per run.",
    )
    p.add_argument("--warmstart-sample", type=int, default=40)
    p.add_argument(
        "--maxmu-arm",
        action="store_true",
        help="add bo_maxmu_select (buy the shortlist's highest posterior mean) "
        "and its bo_random_select control. No model, no GPU, no endpoint -- and "
        "the reference any learnt selector has to beat, so it is worth running "
        "on its own before training anything.",
    )
    p.add_argument(
        "--clm-ckpt",
        default=None,
        help="add bo_clm_select: BO shortlists, trained CLM heads pick "
        "(scripts/clm_train.py). Also adds its controls, bo_random_select and "
        "bo_maxmu_select. Needs the frozen encoder on a GPU; no endpoint.",
    )
    p.add_argument("--shortlist-k", type=int, default=15,
                   help="how many BO-ranked candidates the selector sees (1 = plain BO)")
    p.add_argument(
        "--llm-refresh-every",
        type=int,
        default=0,
        help="rounds between re-proposing the constraint (0 = propose once)",
    )
    p.add_argument("--out", default=None, help="output directory (default results/<dataset stem>)")
    p.add_argument("--value-column", default="pchembl_value")
    p.add_argument("--min-heavy-atoms", type=int, default=10)
    p.add_argument("--max-heavy-atoms", type=int, default=55)
    p.add_argument(
        "--llm-backend",
        choices=["llamacpp", "anthropic", "sara"],
        default="llamacpp",
        help="llamacpp is local, free and deterministic under greedy decoding",
    )
    p.add_argument("--llm-url", default="http://127.0.0.1:8080")
    p.add_argument("--llm-model", default="qwen3.5-4b")
    p.add_argument(
        "--ladder",
        action="store_true",
        help="run the shared arm ladder (molbo.ladder) instead of this script's "
        "own arms, so this tier and the PMO tier are read against the identical "
        "set of methods",
    )
    p.add_argument("--no-llm", action="store_true",
                   help="build the ladder with no endpoint: every model arm is "
                        "skipped and recorded as skipped")
    p.add_argument("--ceiling", action="store_true",
                   help="add bo_policy_oracle, the hindsight ceiling. Reads the "
                        "answer key: a reference line, not a method")
    p.add_argument("--policy-every", type=int, default=1,
                   help="rounds between LLM policy calls; the cost dial for "
                        "bo_policy_llm")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    oracle = LookupOracle.from_dataset(
        args.dataset,
        value_column=args.value_column,
        budget=args.budget,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
    )
    library = oracle.library
    values = oracle.values

    if args.budget > len(library):
        print(
            f"error: budget {args.budget} exceeds the library ({len(library)} compounds). "
            "A run that can buy the whole library measures nothing.",
            file=sys.stderr,
        )
        return 2

    print(
        f"dataset      {args.dataset}\n"
        f"library      {len(library)} compounds "
        f"(heavy atoms {args.min_heavy_atoms}-{args.max_heavy_atoms}), "
        f"{args.value_column} in [{min(values):.2f}, {max(values):.2f}], "
        f"median {np.median(values):.2f}\n"
        f"budget       {args.budget} unique calls   seeds {args.seeds}   n_init {args.n_init}"
    )

    llm_source = None
    if args.llm_arm or args.select_arm or args.warmstart_arm:
        from molbo.llm import AnthropicBackend, LLMConstraintProposer, LlamaCppBackend

        backend = (
            LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
            if args.llm_backend == "llamacpp"
            else AnthropicBackend()
        )
        if args.llm_backend == "llamacpp" and not args.llm_offline:
            ok, detail = backend.health()
            if not ok:
                print(
                    f"error: no llama.cpp server at {args.llm_url} ({detail}).\n"
                    "Start one with:\n"
                    "  llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080 "
                    "--ctx-size 16384 --n-gpu-layers all --reasoning-budget 0",
                    file=sys.stderr,
                )
                return 2
        llm_source = None if not args.llm_arm else LLMConstraintProposer(
            backend=backend,
            cache_dir=args.llm_cache,
            offline=args.llm_offline,
            units=f"{args.value_column} (higher is more potent)",
        )

    arms = default_arms(
        values,
        n_init=args.n_init,
        constraint_fraction=args.constraint_fraction,
        llm_source=llm_source,
    )
    selector = None
    if args.select_arm:
        from molbo.llm import LLMSelector, RandomShortlistSelector

        backend2 = (
            LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
            if args.llm_backend == "llamacpp"
            else AnthropicBackend()
        )
        selector = LLMSelector(backend=backend2, offline=args.llm_offline,
                               units=f"{args.value_column} (higher is more potent)")
        arms.append(ArmConfig(
            "bo_random_select", strategy="bo", acquisition="ei", n_init=args.n_init,
            selector=RandomShortlistSelector(), shortlist_k=args.shortlist_k,
        ))
        arms.append(ArmConfig(
            "bo_llm_select", strategy="bo", acquisition="ei", n_init=args.n_init,
            selector=selector, shortlist_k=args.shortlist_k,
        ))

    clm_selector = None
    if args.maxmu_arm or args.clm_ckpt:
        # Deliberately runnable without --clm-ckpt. `bo_maxmu_select` needs no
        # model at all, and it is the reference a learnt selector has to beat,
        # so it should be measurable before anything is trained. It is also the
        # experiment that says whether the offline shortlist metric predicts
        # campaign AUC: `shortlist_metrics` rewards buying the most potent
        # candidate *this round*, while AUC rewards finding the best compound
        # within the budget, and `bo_random_select` (-0.0229, p=0.0003) shows
        # that departing from EI's ranking can cost real AUC.
        from molbo.clm import MaxMeanSelector
        from molbo.llm import RandomShortlistSelector

        if not any(a.name == "bo_random_select" for a in arms):
            arms.append(ArmConfig(
                "bo_random_select", strategy="bo", acquisition="ei", n_init=args.n_init,
                selector=RandomShortlistSelector(), shortlist_k=args.shortlist_k,
            ))
        arms.append(ArmConfig(
            "bo_maxmu_select", strategy="bo", acquisition="ei", n_init=args.n_init,
            selector=MaxMeanSelector(), shortlist_k=args.shortlist_k,
        ))
        if args.ceiling and hasattr(oracle, "_table"):
            from molbo.clm import OracleShortlistSelector

            arms.append(ArmConfig(
                "bo_oracle_select", strategy="bo", acquisition="ei", n_init=args.n_init,
                selector=OracleShortlistSelector(values=dict(oracle._table)),
                shortlist_k=args.shortlist_k,
            ))

    if args.clm_ckpt:
        from molbo.clm import CLMSelector

        clm_selector = CLMSelector.from_checkpoint(args.clm_ckpt)
        arms.append(ArmConfig(
            "bo_clm_select", strategy="bo", acquisition="ei", n_init=args.n_init,
            selector=clm_selector, shortlist_k=args.shortlist_k,
        ))

    warmstart = None
    if args.warmstart_arm:
        from molbo.llm import LLMInitSelector

        backend3 = (
            LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
            if args.llm_backend == "llamacpp"
            else AnthropicBackend()
        )
        warmstart = LLMInitSelector(
            backend=backend3, offline=args.llm_offline,
            sample_size=args.warmstart_sample,
            units=f"{args.value_column} (higher is more potent)",
        )
        from molbo.llm import DiverseInitSelector

        arms.append(ArmConfig(
            "bo_diverse_warmstart", strategy="bo", acquisition="ei",
            n_init=args.n_init,
            init_source=DiverseInitSelector(sample_size=args.warmstart_sample),
        ))
        arms.append(ArmConfig(
            "bo_llm_warmstart", strategy="bo", acquisition="ei",
            n_init=args.n_init, init_source=warmstart,
        ))

    if llm_source is not None and args.llm_refresh_every and not args.ladder:
        for a in arms:
            if a.name == "bo_llm_constraint":
                a.constraint_every = args.llm_refresh_every
    ladder_skipped: list[str] = []
    if args.ladder:
        from molbo.ladder import LadderSpec, build_ladder, skipped_arms

        ladder_backend = None
        if not args.no_llm:
            if args.llm_backend == "sara":
                from molbo.agentic.sara import SaraBackend
                from molbo.llm import LLMUnavailable

                try:
                    ladder_backend = SaraBackend.from_env()
                except LLMUnavailable as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
            elif args.llm_backend == "anthropic":
                from molbo.llm import AnthropicBackend

                ladder_backend = AnthropicBackend()
            else:
                from molbo.llm import LlamaCppBackend

                ladder_backend = LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
            ok, detail = ladder_backend.health()
            if not ok:
                print(
                    f"error: no model endpoint ({detail}). Start one, or pass "
                    "--no-llm to run the non-model half of the ladder.",
                    file=sys.stderr,
                )
                return 2

        spec = LadderSpec(
            n_init=args.n_init,
            shortlist_k=args.shortlist_k,
            units=f"{args.value_column} (higher is more potent)",
            policy_every=args.policy_every,
        )
        # The ceiling is affordable on this tier and not on PMO: a LookupOracle
        # already holds every value, so `bo_policy_oracle` costs no extra calls.
        arms = build_ladder(
            spec,
            backend=ladder_backend,
            cache_root=args.llm_cache.rstrip("/").rsplit("/", 1)[0]
            if "/" in args.llm_cache else "data/cache",
            values=values if args.ceiling else None,
            # Read off the oracle rather than asserted: `OraclePolicy` picks the
            # acquisition whose top candidate has the best *true* value, so a
            # hardcoded `True` against a minimising oracle would draw the
            # hindsight ceiling through the worst molecules and report it as a
            # bound on every method.
            maximize=bool(getattr(oracle, "maximize", True)),
        )
        ladder_skipped = skipped_arms(
            ladder_backend is not None, ("none", "bo", "llm", "bo+llm", "agentic")
        )
        if ladder_skipped:
            print(f"skipped (no endpoint): {', '.join(ladder_skipped)}")

    if args.arms:
        wanted = set(args.arms)
        unknown = wanted - {a.name for a in arms}
        if unknown:
            print(
                f"error: unknown arm(s) {sorted(unknown)}; "
                f"available: {sorted(a.name for a in arms)}",
                file=sys.stderr,
            )
            return 2
        arms = [a for a in arms if a.name in wanted]

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"arms         {', '.join(a.name for a in arms)}\n")

    t0 = time.perf_counter()
    metrics, traces = run_sweep(
        oracle, arms, seeds, budget=args.budget, k=args.k, library=library
    )
    elapsed = time.perf_counter() - t0

    summary = summarize_sweep(metrics, reference="bo_ei")
    out_dir = Path(args.out or f"results/{Path(args.dataset).stem}")
    meta = {
        "dataset": str(args.dataset),
        "library_size": len(library),
        "budget": args.budget,
        "seeds": seeds,
        "n_init": args.n_init,
        "k": args.k,
        "constraint_fraction": args.constraint_fraction,
        "best_possible": oracle.best_value,
        "oracle": oracle.describe(),
        "wall_seconds": round(elapsed, 1),
        # Printed to stdout is not recorded. A reader of summary.json must be
        # able to tell "this arm lost" from "this arm never ran", and stdout is
        # gone by the time anyone opens the file.
        "skipped_arms": ladder_skipped,
        "arms_run": [a.name for a in arms],
    }
    save_sweep(out_dir, metrics, traces, summary, meta=meta)

    print(f"\n{'arm':24s} {'AUC':>18s} {'best pIC50':>16s} {'regret':>8s} {'top1%':>7s}")
    print("-" * 78)
    by_arm: dict[str, list] = {}
    for m in metrics:
        by_arm.setdefault(m.arm, []).append(m)
    for name, s in summary["summaries"].items():
        rows = by_arm[name]
        regret = np.mean([m.simple_regret for m in rows if m.simple_regret is not None])
        recall = np.mean([m.top1pct_recall for m in rows if m.top1pct_recall is not None])
        print(
            f"{name:24s} {s.auc_mean:8.4f} +/- {s.auc_sem:.4f} "
            f"{s.best_mean:9.3f} +/- {s.best_sem:.3f} {regret:8.3f} {recall:7.3f}"
        )

    print(f"\npaired vs {summary['reference']} (Wilcoxon signed-rank on per-seed AUC, n={len(seeds)})")
    print("-" * 78)
    for name, cmp in summary["comparisons"].items():
        if "error" in cmp:
            print(f"{name:24s} {cmp['error']}")
            continue
        stars = "***" if cmp["p_value"] < 0.001 else "**" if cmp["p_value"] < 0.01 else "*" if cmp["p_value"] < 0.05 else "ns"
        print(
            f"{name:24s} dAUC {cmp['mean_diff']:+.4f}  "
            f"dz {cmp['cohens_dz']:+.2f}  p {cmp['p_value']:.4f}  {stars}"
        )

    if llm_source is not None:
        n_dropped = sum(1 for e in llm_source.log if e.get("dropped"))
        n_none = sum(1 for e in llm_source.log if e.get("constraint") is None)
        print(
            f"\nLLM ({args.llm_backend}): {llm_source.calls} call(s), "
            f"{llm_source.cache_hits} cache hit(s), {llm_source.failures} failure(s)\n"
            f"     {n_dropped}/{len(llm_source.log)} proposals had a clause dropped; "
            f"{n_none} yielded no usable constraint"
        )
        # The proposals themselves are the arm's primary artifact: what the
        # model actually claimed, and what survived compilation.
        (out_dir / "llm_log.json").write_text(
            json.dumps(llm_source.log, indent=2), encoding="utf-8"
        )

    if warmstart is not None:
        print(
            f"\nwarm start: {warmstart.calls} call(s), "
            f"{warmstart.cache_hits} cache hit(s), {warmstart.failures} failure(s)"
        )
        (out_dir / "warmstart_log.json").write_text(
            json.dumps(warmstart.log, indent=2), encoding="utf-8"
        )

    if selector is not None:
        n_dec = len(selector.ranks)
        print(
            f"\nselector: {selector.calls} call(s), {selector.cache_hits} cache hit(s), "
            f"{selector.failures} failure(s), {selector.deferred} deferred"
        )
        if n_dec:
            k = args.shortlist_k
            ranks = np.asarray(selector.ranks)
            hist = [int((ranks == i).sum()) for i in range(k)]
            # A mean near (k-1)/2 looks like random choice, but the *shape*
            # is what tells you why. A model picking by list position rather
            # than by chemistry piles up at the ends.
            print(
                f"     mean chosen rank {ranks.mean():.2f} of {k} "
                f"(0 = agrees with BO; {(k - 1) / 2:.1f} = uniform)\n"
                f"     picked BO's top choice {100 * (ranks == 0).mean():.1f}% "
                f"(uniform would be {100 / k:.1f}%)\n"
                f"     rank histogram {hist}"
            )
        (out_dir / "selector_ranks.json").write_text(
            json.dumps(
                {
                    "shortlist_k": args.shortlist_k,
                    "ranks": [int(r) for r in selector.ranks],
                    "calls": selector.calls,
                    "cache_hits": selector.cache_hits,
                    "failures": selector.failures,
                    "deferred": selector.deferred,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if clm_selector is not None and clm_selector.ranks:
        ranks = np.asarray(clm_selector.ranks)
        k = args.shortlist_k
        print(
            f"\nclm selector: {clm_selector.calls} pick(s), "
            f"mean chosen rank {ranks.mean():.2f} of {k}, "
            f"picked BO's top choice {100 * (ranks == 0).mean():.1f}%\n"
            f"     rank histogram {[int((ranks == i).sum()) for i in range(k)]}"
        )
        (out_dir / "clm_selector.json").write_text(
            json.dumps(
                {"checkpoint": args.clm_ckpt, "encoder": clm_selector.encoder.key(),
                 "shortlist_k": k, "ranks": [int(r) for r in ranks],
                 "probs": clm_selector.probs},
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"\nwrote {out_dir}/  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

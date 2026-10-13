"""Run arms on PMO tasks over the ZINC-250k pool, at a fixed unique-call budget.

    python scripts/run_pmo_experiment.py --tasks qed albuterol_similarity --seeds 5

This is the comparability tier. The primary ChEMBL oracle is one we built, which
is exactly what reviewers distrust; PMO's tasks come with 25 published baselines.

What is and is not comparable
-----------------------------
The search space here is the **ZINC-250k pool**, so the directly comparable PMO
baselines are the ones that also search a fixed library -- ``screening`` (random
sampling from the pool) and ``mol_pal`` (BO over the pool). The generative
baselines (REINVENT, GraphGA, SMILES-GA, ...) search unrestricted chemical
space and will generally score higher for that reason alone; quote them as
context, not as a like-for-like comparison.

**Budget.** PMO's published tables use 10,000 oracle calls. This script
defaults to 1,000, which is the regime an expensive oracle actually lives in
and the setting recent work uses. The two numbers are *not* interchangeable and
the budget is written into every output file. An exact GP is also O(n^3) per
fit, so 10,000 observations is not reachable with this surrogate -- state the
budget whenever the number is quoted, and note that 1,000 is the harder
setting.

Cost control: the GP scores ``--candidate-subset`` randomly drawn unevaluated
molecules per round rather than all 250k. The search space is still the whole
pool; only the per-round shortlist is sampled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from molbo.campaign import ArmConfig, run_campaign  # noqa: E402
from molbo.constraints import LibraryContext  # noqa: E402
from molbo.metrics import auc_top_k, compare_arms, summarize_arm, top_auc  # noqa: E402
from molbo.oracles.pmo import PMO_TASKS, PMOOracle, probe_oracle  # noqa: E402
from molbo.chem import FINGERPRINT_VERSION  # noqa: E402
from molbo.surrogate import PackedFingerprints  # noqa: E402

ZINC_TAB = Path("data/cache/tdc/zinc.tab")
FP_CACHE = Path("data/cache/zinc_fp.npz")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tasks", nargs="+", default=["qed", "albuterol_similarity"])
    p.add_argument("--budget", type=int, default=1000, help="unique oracle calls")
    p.add_argument("--seeds", type=int, default=5, help="PMO convention is 5 runs")
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--candidate-subset", type=int, default=2000)
    p.add_argument("--pool-size", type=int, default=None, help="subsample ZINC (default: all)")
    p.add_argument("--k", type=int, default=10, help="top-k for the AUC metric")
    p.add_argument("--out", default="results/pmo")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument(
        "--warmstart-arms",
        action="store_true",
        help="add bo_llm_warmstart and its MaxMin-diversity control. One model "
        "call per run, so this is affordable here; the selection arm (one call "
        "per evaluation) is not, at a 1000-call budget.",
    )
    p.add_argument("--warmstart-sample", type=int, default=40)
    p.add_argument(
        "--constraint-arm",
        action="store_true",
        help="add bo_llm_constraint. One model call per run. The substructure "
        "menu is mined from a subsample of the pool but counted on all of it.",
    )
    p.add_argument("--menu-mine-sample", type=int, default=3000)
    p.add_argument("--llm-url", default="http://127.0.0.1:8080")
    p.add_argument("--llm-model", default="qwen3.5-4b")
    p.add_argument(
        "--ladder",
        action="store_true",
        help="run the shared arm ladder (molbo.ladder) instead of this script's "
        "own two arms, so the PMO tier and the ChEMBL tier are read against the "
        "identical set of methods",
    )
    p.add_argument(
        "--llm-backend",
        choices=["llamacpp", "sara"],
        default="llamacpp",
        help="sara reads SARA_BASE_URL / SARA_MODEL / SARA_API_KEY",
    )
    p.add_argument("--shortlist-k", type=int, default=15,
                   help="candidates offered to a selector; k=1 collapses to the "
                        "arm it is built on")
    p.add_argument("--policy-every", type=int, default=1,
                   help="rounds between LLM policy calls; the cost dial for "
                        "bo_policy_llm, which can otherwise ask once per call")
    p.add_argument("--no-llm", action="store_true",
                   help="build the ladder with no endpoint: every model arm is "
                        "skipped and recorded as skipped")
    return p.parse_args(argv)


def load_pool(pool_size: int | None):
    import pandas as pd

    if not ZINC_TAB.exists():
        print(
            f"error: {ZINC_TAB} not found. Fetch it once with:\n"
            "  python -c \"from tdc.generation import MolGen; "
            "MolGen(name='ZINC', path='data/cache/tdc')\"",
            file=sys.stderr,
        )
        raise SystemExit(2)
    smiles = pd.read_csv(ZINC_TAB, sep="\t")["smiles"].tolist()
    if pool_size is not None and pool_size < len(smiles):
        # Fixed seed: the pool must be identical across tasks and arms, or the
        # runs are not comparable to each other, never mind to PMO.
        rng = np.random.default_rng(12345)
        idx = np.sort(rng.choice(len(smiles), size=pool_size, replace=False))
        smiles = [smiles[i] for i in idx]
    return smiles


def load_fingerprints(smiles, pool_size):
    """Fingerprint the pool once and keep it packed on disk."""
    tag = "full" if pool_size is None else str(pool_size)
    # The fingerprint recipe is part of the filename. Without it, changing the
    # representation would silently reuse a stale pool -- the one cache in this
    # project not keyed by content.
    cache = FP_CACHE.with_name(f"zinc_fp_{tag}_{FINGERPRINT_VERSION}.npz")
    if cache.exists():
        z = np.load(cache)
        fp = PackedFingerprints.__new__(PackedFingerprints)
        fp.n_bits, fp.packed, fp.valid = int(z["n_bits"]), z["packed"], z["valid"]
        print(f"pool fingerprints: {cache} ({fp.nbytes / 1e6:.0f} MB, cached)")
        return fp

    t = time.perf_counter()
    print(f"fingerprinting {len(smiles):,} molecules (one-off)...", flush=True)
    fp = PackedFingerprints(smiles, progress=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, packed=fp.packed, valid=fp.valid, n_bits=fp.n_bits)
    print(f"  done in {time.perf_counter() - t:.0f}s -> {cache} ({fp.nbytes / 1e6:.0f} MB)")
    return fp


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_tasks:
        print("\n".join(PMO_TASKS))
        return 0

    unknown = set(args.tasks) - set(PMO_TASKS)
    if unknown:
        print(f"error: unknown task(s) {sorted(unknown)}", file=sys.stderr)
        return 2

    smiles = load_pool(args.pool_size)
    fps = load_fingerprints(smiles, args.pool_size)
    # Lazy: molecules and descriptors are parsed on first use, so this is
    # correct for the constraint arm and free for the arms that never ask.
    ctx = LibraryContext(smiles=smiles)
    seeds = list(range(args.seeds))

    arms = [
        ArmConfig(
            "random", strategy="random", n_init=args.n_init, batch_size=args.batch_size
        ),
        ArmConfig(
            "bo_ei",
            strategy="bo",
            acquisition="ei",
            n_init=args.n_init,
            batch_size=args.batch_size,
            candidate_subset=args.candidate_subset,
        ),
    ]

    constraint = None
    if args.constraint_arm:
        from molbo.llm import LLMConstraintProposer, LlamaCppBackend

        backend = LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
        ok, detail = backend.health()
        if not ok:
            print(f"error: no llama.cpp server at {args.llm_url} ({detail})", file=sys.stderr)
            return 2
        constraint = LLMConstraintProposer(
            backend=backend,
            cache_dir="data/cache/llm_pmo",
            units="a task-specific desirability score in [0, 1] (higher is better)",
            menu_kwargs={"mine_sample": args.menu_mine_sample},
        )
        arms.append(ArmConfig(
            "bo_llm_constraint", strategy="bo", acquisition="ei", n_init=args.n_init,
            batch_size=args.batch_size, candidate_subset=args.candidate_subset,
            constraint_source=constraint, min_candidates=200,
        ))

    warmstart = None
    if args.warmstart_arms:
        from molbo.llm import DiverseInitSelector, LlamaCppBackend, LLMInitSelector

        backend = LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
        ok, detail = backend.health()
        if not ok:
            print(f"error: no llama.cpp server at {args.llm_url} ({detail})", file=sys.stderr)
            return 2
        common = dict(strategy="bo", acquisition="ei", n_init=args.n_init,
                      batch_size=args.batch_size, candidate_subset=args.candidate_subset)
        arms.append(ArmConfig(
            "bo_diverse_warmstart",
            init_source=DiverseInitSelector(sample_size=args.warmstart_sample), **common,
        ))
        # The objective text is task-specific here, unlike the ChEMBL tier: a
        # PMO score is an arbitrary [0, 1] desirability, not a potency.
        warmstart = LLMInitSelector(
            backend=backend, sample_size=args.warmstart_sample,
            cache_dir="data/cache/llm_init_pmo",
            units="a task-specific desirability score in [0, 1] (higher is better)",
        )
        arms.append(ArmConfig("bo_llm_warmstart", init_source=warmstart, **common))

    skipped: list[str] = []
    if args.ladder:
        from molbo.ladder import LadderSpec, build_ladder, skipped_arms

        backend = None
        if not args.no_llm:
            if args.llm_backend == "sara":
                from molbo.agentic.sara import SaraBackend
                from molbo.llm import LLMUnavailable

                try:
                    backend = SaraBackend.from_env()
                except LLMUnavailable as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
            else:
                from molbo.llm import LlamaCppBackend

                backend = LlamaCppBackend(base_url=args.llm_url, model=args.llm_model)
            ok, detail = backend.health()
            if not ok:
                print(
                    f"error: no model endpoint ({detail}). "
                    "Start one, or pass --no-llm to run the non-model half of "
                    "the ladder.",
                    file=sys.stderr,
                )
                return 2

        spec = LadderSpec(
            n_init=args.n_init,
            batch_size=args.batch_size,
            candidate_subset=args.candidate_subset,
            shortlist_k=args.shortlist_k,
            # A PMO score is an arbitrary [0, 1] desirability, not a potency.
            # Telling the model it is reading pIC50 would be a lie that changes
            # its answers.
            units="a task-specific desirability score in [0, 1] (higher is better)",
            policy_every=args.policy_every,
            constraint_min_candidates=200,
            # Without this the menu is mined from all 249,455 ZINC molecules.
            menu_mine_sample=args.menu_mine_sample,
        )
        # No ceiling on this tier: bo_policy_oracle needs the true value of
        # every pool molecule, and a PMO oracle is a function call rather than a
        # lookup table, so drawing that reference line would cost 250k calls per
        # task -- far more than the experiment it annotates.
        arms = build_ladder(spec, backend=backend, cache_root="data/cache/pmo", values=None)
        skipped = skipped_arms(
            backend is not None, ("none", "bo", "llm", "bo+llm", "agentic")
        )
        if skipped:
            print(f"skipped (no endpoint): {', '.join(skipped)}")

    print(
        f"\npool        {len(smiles):,} molecules (ZINC-250k"
        f"{'' if args.pool_size is None else f', subsampled to {args.pool_size:,}'})\n"
        f"budget      {args.budget} unique calls   seeds {args.seeds}   "
        f"batch {args.batch_size}\n"
        f"candidates  {args.candidate_subset:,} scored per round\n"
        f"tasks       {', '.join(args.tasks)}\n"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    everything: dict[str, dict] = {}

    def build_payload() -> dict:
        """The results file as of right now.

        A function rather than a literal because it is written after every task,
        not only at the end: an unloadable oracle used to take every completed
        task down with it.
        """
        return {
            "benchmark": "PMO",
            "pool": "ZINC-250k" if args.pool_size is None else f"ZINC subsample {args.pool_size}",
            "pool_size": len(smiles),
            "budget": args.budget,
            "seeds": seeds,
            "n_init": args.n_init,
            "batch_size": args.batch_size,
            "candidate_subset": args.candidate_subset,
            "k": args.k,
            "metric": "AUC of top-k mean vs cumulative unique oracle calls",
            "comparable_pmo_baselines": ["screening", "mol_pal"],
            "note": (
                "PMO's published tables use a 10,000-call budget; this run uses "
                f"{args.budget}. The numbers are not interchangeable."
            ),
            "requested_tasks": list(args.tasks),
            "unavailable_tasks": dict(unavailable),
            "flat_tasks": dict(flat_tasks),
            # Printed to stdout is not recorded. A reader must be able to tell
            # "this arm lost" from "this arm never ran", and stdout is gone by
            # the time anyone opens the file.
            "skipped_arms": list(skipped),
            "arms_run": [a.name for a in arms],
            "warmstart_llm_calls": (warmstart.calls if warmstart else 0),
            "constraint_llm_calls": (constraint.calls if constraint else 0),
            # `--ladder` replaces this script's own arms, so `warmstart` and
            # `constraint` above are the discarded objects and read 0 however
            # many calls the ladder's own sources made. Count them off the arms
            # that actually ran.
            "llm_calls": _llm_calls(),
            "results": everything,
        }

    def _llm_calls() -> dict[str, int]:
        out: dict[str, int] = {}
        for a in arms:
            for kind in ("constraint_source", "selector", "policy_source", "init_source"):
                n = getattr(getattr(a, kind, None), "calls", None)
                if n:
                    out[f"{a.name}.{kind}"] = int(n)
        return out

    unavailable: dict[str, str] = {}
    flat_tasks: dict[str, str] = {}
    for task in args.tasks:
        print(f"=== {task} ===", flush=True)
        status, detail = probe_oracle(task)
        if status == "broken":
            # Probed before any budget is spent. `drd2` and `gsk3b` load fine on
            # a current sklearn and then score every molecule identically, so
            # without this the run burns its full budget producing 0.0000 for
            # every arm -- which reads as a tie rather than a broken oracle.
            unavailable[task] = detail
            print(f"  SKIPPED - oracle is unusable here: {detail}", flush=True)
            continue
        if status == "flat":
            # A real but impossible task (valsartan_smarts). Run it -- dropping
            # it would quietly change the benchmark -- but record it, because a
            # mean including it is not comparable to one that excludes it.
            flat_tasks[task] = detail
            print(f"  NOTE - degenerate objective: {detail}", flush=True)
        try:
            oracle = PMOOracle(task, budget=args.budget)
        except Exception as exc:
            # TDC ships pickled scikit-learn models; `jnk3` was saved before
            # sklearn 1.3 added `missing_go_to_left` to the tree node dtype, so
            # it raises on a current sklearn. One unloadable oracle must not
            # cost the other 22 tasks their results -- and it must be recorded,
            # because a task silently missing from the table is indistinguishable
            # from a task where every arm tied.
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            unavailable[task] = reason
            print(f"  SKIPPED - oracle would not load: {reason}", flush=True)
            continue
        per_arm: dict[str, list[dict]] = {}
        for arm in arms:
            t0 = time.perf_counter()
            rows = []
            traces_for_arm = []
            policy_traces: list[dict] = []
            for seed in seeds:
                res = run_campaign(
                    oracle, arm, seed, budget=args.budget,
                    library=smiles, ctx=ctx, X=fps, kernel=False,
                )
                traces_for_arm.append(res)
                scored = res.scored
                buf = {s: (float(v), i + 1)
                       for i, (s, v) in enumerate(zip(res.smiles, res.values))
                       if v is not None}
                rows.append(
                    {
                        "seed": seed,
                        # PMO scores are already in [0, 1] and higher-is-better,
                        # so unlike the ChEMBL tier they go in unscaled.
                        "auc": auc_top_k(scored, k=args.k, budget=args.budget),
                        "auc_pmo": top_auc(buf, top_n=args.k, freq_log=100,
                                           max_oracle_calls=args.budget),
                        "top1": max(scored) if scored else 0.0,
                        "top10": float(np.mean(sorted(scored, reverse=True)[: args.k]))
                        if scored else 0.0,
                        "n_calls": res.n_calls,
                        "wall": res.wall_seconds,
                    }
                )
                # What the policy actually changed, round by round. This is the
                # point of the agentic arms: an arm that switched once and an
                # arm that switched every round can post the same AUC, and only
                # the trajectory tells them apart.
                if res.policy_log:
                    policy_traces.append({"seed": seed, "log": res.policy_log})
            per_arm[arm.name] = rows
            if policy_traces:
                per_arm[f"_policy_{arm.name}"] = policy_traces
            # A constraint that matches nothing degrades to unconstrained and
            # would otherwise be indistinguishable from plain BO in the output.
            modes: dict[str, int] = {}
            allowed: list[float] = []
            for t in traces_for_arm:
                for e in t.constraint_log:
                    modes[e.get("mode", "?")] = modes.get(e.get("mode", "?"), 0) + 1
                    if e.get("fraction_allowed") is not None:
                        allowed.append(e["fraction_allowed"])
            if modes and arm.constraint_source is not None:
                frac = f", kept {100 * np.mean(allowed):.1f}% of the pool" if allowed else ""
                print(f"    constraint modes {modes}{frac}")
            s = summarize_arm(arm.name, [r["auc"] for r in rows], [r["top1"] for r in rows])
            print(
                f"  {arm.name:10s} AUC(top-{args.k}) {s.auc_mean:.4f} +/- {s.auc_sem:.4f}"
                f"   top-1 {s.best_mean:.4f}"
                f"   top-{args.k} {np.mean([r['top10'] for r in rows]):.4f}"
                f"   ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

        for a in ("bo_llm_constraint", "bo_diverse_warmstart", "bo_llm_warmstart"):
            if a in per_arm:
                c = compare_arms([r["auc"] for r in per_arm[a]],
                                 [r["auc"] for r in per_arm["bo_ei"]], paired=True)
                print(f"  {a:22s} vs bo_ei: dAUC {c['mean_diff']:+.4f}  p {c['p_value']:.4f}")
                per_arm[f"_cmp_{a}"] = c
        if {"bo_llm_warmstart", "bo_diverse_warmstart"} <= per_arm.keys():
            c = compare_arms([r["auc"] for r in per_arm["bo_llm_warmstart"]],
                             [r["auc"] for r in per_arm["bo_diverse_warmstart"]], paired=True)
            print(f"  {'LLM vs diversity':22s} warm start: dAUC {c['mean_diff']:+.4f}  p {c['p_value']:.4f}")
            per_arm["_cmp_llm_vs_diverse"] = c

        if {"bo_ei", "random"} <= per_arm.keys():
            cmp = compare_arms(
                [r["auc"] for r in per_arm["bo_ei"]],
                [r["auc"] for r in per_arm["random"]],
                paired=True,
            )
            print(
                f"  bo_ei vs random: dAUC {cmp['mean_diff']:+.4f}  p {cmp['p_value']:.4f}"
            )
            per_arm["_comparison"] = cmp
        everything[task] = per_arm
        # Checkpoint after every task. A crash 10 tasks in used to discard
        # all ten; the file is small and rewriting it costs nothing.
        (out_dir / "pmo_results.json").write_text(
            json.dumps(build_payload(), indent=2), encoding="utf-8")

    (out_dir / "pmo_results.json").write_text(
        json.dumps(build_payload(), indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}/pmo_results.json")
    if unavailable:
        print("\nunavailable oracles (recorded in the results file):")
        for t, why in unavailable.items():
            print(f"  {t:26s} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

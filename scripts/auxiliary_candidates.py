"""Does a cold target have anything to select from? -- gate 1 for an agentic selector.

    python scripts/auxiliary_candidates.py

The proposal this answers: for a protein with no assay of its own, have an agent
search the public databases for relatives, pull their measured history, and run
the multi-task arm against that. Section 34 says the surrogate half of that works;
section 32 says the auxiliaries have to clear Spearman ~0.5 to carry anything.

Neither says the *candidates exist*. That is this script's question, and it is a
question about BindingDB's geometry rather than about any agent:

    for each target we can score, how many other targets are co-measured with it
    on enough shared compounds to be usable -- and how many of those are outside
    its own sequence family?

Co-measurement is the binding constraint and it is easy to miss. Section 17 found
no transfer on *disjoint* per-target libraries (+0.008 +- 0.007 after the section 30
audit), and section 18 traced that to library geometry rather than task
correlation. Every positive result in this package sits on a fully-observed
block: the same compounds run against every target in the set. An internet
search returns related targets; it does not return co-measured ones. If a cold
target's only co-measured partners are the three paralogues already in its
registry row, then the agent has nothing to choose and the ceiling on the whole
idea is what the database happens to hold, not how well anything reasons.

So the output is a count, per target, of the out-of-family candidates -- the
selectable set. An empty one closes the gate; a large one means selection is a
real decision and gate 2 (can an agent rank that set before seeing any labels?)
is worth running.

Sizes are reported raw *and* after the censoring filter, because censoring costs
42% of JAK and 77% of CYP and a raw intersection clearing the 160-compound floor
routinely does not survive it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from find_dense_blocks import MIN_POOL, incidence, uniprot_names  # noqa: E402


def columns(M):
    """Boolean membership column per target, materialised on demand.

    numpy's AND over a few hundred thousand bits is far faster than scipy's
    sparse one, which is why `search()` in find_dense_blocks densifies too. It
    densifies *everything* up front, which is fine at the 460 targets its filter
    leaves but is a gigabyte once the filter is relaxed -- and relaxing it is the
    control this script has to run, since a candidate shut out for having 300
    compounds of its own could still share 160 with the primary. Only the
    candidate pools are ever touched, and they are small, so cache per column.
    """
    Mc = M.tocsc()
    cache: dict[int, np.ndarray] = {}

    def col(j):
        if j not in cache:
            cache[j] = np.asarray(Mc[:, j].todense()).ravel().astype(bool)
        return cache[j]

    return col


def grow(cols, primary, pool, size):
    """Greedily add from ``pool`` the auxiliaries that keep the most compounds.

    Returns ``(members, n_common)`` with ``primary`` first. Greedy for the reason
    the block search is greedy -- the exact choice is C(460, 3) per primary and
    the answer only has to be a lower bound on what is available.
    """
    cur, m = [primary], cols(primary).copy()
    while len(cur) < size:
        best, best_n, best_m = None, 0, None
        for j in pool:
            if j in cur:
                continue
            mj = m & cols(j)
            n = int(mj.sum())
            if n > best_n:
                best, best_n, best_m = j, n, mj
        if best is None:
            break
        cur.append(best)
        m = best_m
    return cur, int(m.sum())


def clean_size(accessions, pile_frac):
    """Compounds left in a block after the censoring filter.

    The quantity a campaign would actually see. Blocks are keyed by accession
    here rather than by gene name, so this shares `build_block`'s cache with
    nothing else and cannot collide with the registry's own blocks.
    """
    from dense_block_h1 import build_block, drop_censored

    mapping = {a: a for a in accessions}
    block, _ = build_block(mapping)
    present = [a for a in accessions if a in block.columns]
    if len(present) < len(accessions):
        return 0
    clean, _ = drop_censored(block, present, pile_frac=pile_frac, verbose=False)
    return len(clean)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-per-target", type=int, default=400,
                   help="same filter the block scan uses")
    p.add_argument("--size", type=int, default=4, help="targets per block")
    p.add_argument("--pile-frac", type=float, default=0.02)
    p.add_argument("--no-names", action="store_true",
                   help="skip the UniProt lookup; print accessions")
    p.add_argument("--out", default="results/auxiliary_candidates.json")
    args = p.parse_args(argv)

    from dense_block_campaign import FAMILIES

    M, targets, n_pairs = incidence(min_per_target=args.min_per_target)
    idx = {a: j for j, a in enumerate(targets)}
    fam_of = {acc: fam for fam, m in FAMILIES.items() for acc in m}
    cols = columns(M)
    C = (M.T @ M).toarray()
    n_aux = args.size - 1

    print(f"{n_pairs:,} unique (compound, target) pairs; {len(targets)} targets "
          f"with >= {args.min_per_target} compounds\n")

    # Only the families a campaign can actually run on -- the three the registry
    # lists as too small have no primary worth asking the question about.
    from find_dense_blocks import family_properties
    prop = family_properties(args.pile_frac)
    runnable = [f for f, q in prop.items()
                if q["runnable"] and q["coverage_at_H50"] <= 0.9]

    rows = []
    for fam in runnable:
        for acc, gene in FAMILIES[fam].items():
            if acc not in idx:
                # Below the min-per-target filter: it is in the registry because
                # the block is dense, not because the target is well covered.
                rows.append({"family": fam, "accession": acc, "gene": gene,
                             "in_scan": False})
                continue
            t = idx[acc]
            # Candidates are targets sharing enough compounds with the primary to
            # clear the pool floor pairwise. This is an upper bound on the set a
            # 4-block can be built from: adding members only shrinks the
            # intersection, never grows it.
            share = C[t]
            cand = [j for j in np.where(share >= MIN_POOL)[0]
                    if j != t]
            out = [j for j in cand if fam_of.get(targets[j]) != fam]
            fam_members = [j for j in cand if fam_of.get(targets[j]) == fam]
            best_out, n_out_raw = (grow(cols, t, out, args.size)
                                   if len(out) >= n_aux else ([t], 0))
            best_any, n_any_raw = grow(cols, t, cand, args.size) \
                if len(cand) >= n_aux else ([t], 0)
            rows.append({
                "family": fam, "accession": acc, "gene": gene, "in_scan": True,
                "n_candidates": len(cand),
                "n_in_family": len(fam_members),
                "n_out_of_family": len(out),
                "n_out_ge400": int(sum(share[j] >= 400 for j in out)),
                "best_out_of_family": [targets[j] for j in best_out[1:]],
                "best_out_raw": n_out_raw,
                "best_any": [targets[j] for j in best_any[1:]],
                "best_any_raw": n_any_raw,
            })

    printed = sorted({a for r in rows for a in r.get("best_out_of_family", [])})
    names = {} if args.no_names else uniprot_names(printed)

    def label(acc):
        return names.get(acc, acc)

    print("[1] co-measured candidates per target, at the 160-compound pool floor")
    print(f"    {'target':<10}{'cand':>6}{'in-fam':>8}{'out':>6}{'out>=400':>10}"
          f"{'best-3 out':>12}  members")
    for r in rows:
        if not r["in_scan"]:
            print(f"    {r['gene']:<10}  not in the scan "
                  f"(< {args.min_per_target} compounds of its own)")
            continue
        members = " ".join(label(a) for a in r["best_out_of_family"]) or "-"
        print(f"    {r['gene']:<10}{r['n_candidates']:>6}{r['n_in_family']:>8}"
              f"{r['n_out_of_family']:>6}{r['n_out_ge400']:>10}"
              f"{r['best_out_raw']:>12,}  {members}")

    scanned = [r for r in rows if r["in_scan"]]
    if not scanned:
        print("\n    no registry target clears the scan filter")
        return 0

    print("\n[2] is there a selection to make?")
    outs = [r["n_out_of_family"] for r in scanned]
    none = sum(o == 0 for o in outs)
    print(f"    {len(scanned)} targets scored; out-of-family candidates "
          f"median {int(np.median(outs))}, range {min(outs)}-{max(outs)}")
    print(f"    {none}/{len(scanned)} have no out-of-family candidate at all")
    buildable = [r for r in scanned if r["best_out_raw"] >= MIN_POOL]
    print(f"    {len(buildable)}/{len(scanned)} can form a {args.size}-target "
          f"block from out-of-family targets alone (raw >= {MIN_POOL})")

    print("\n[3] what survives censoring -- the size a campaign sees")
    for r in buildable:
        accs = [r["accession"], *r["best_out_of_family"]]
        n = clean_size(accs, args.pile_frac)
        r["best_out_clean"] = n
        mark = " " if n >= MIN_POOL else "x"
        print(f"  {mark} {r['gene']:<10} raw {r['best_out_raw']:>6,} -> clean "
              f"{n:>6,}   {' '.join(label(a) for a in r['best_out_of_family'])}")
    usable = [r for r in buildable if r.get("best_out_clean", 0) >= MIN_POOL]
    print(f"    {len(usable)}/{len(scanned)} targets have a usable "
          "out-of-family block after censoring")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"min_per_target": args.min_per_target,
                               "size": args.size, "pile_frac": args.pile_frac,
                               "n_targets": len(targets), "targets": rows},
                              indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

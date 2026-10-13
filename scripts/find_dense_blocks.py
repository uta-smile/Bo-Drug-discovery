"""Which protein families does BindingDB actually hold a dense block for?

    python scripts/find_dense_blocks.py --out results/dense_blocks_scan.json

Sections 18-25 rest on two blocks, JAK and CYP, and both were chosen by hand.
That is a weak footing for a claim of the form "this works on co-measured
panels": two families picked by someone who knew what they were looking for
cannot distinguish "works on panels" from "works on these two".

This script removes the hand-picking. A **dense block** is a set of targets and
the compounds measured against *every* member of the set -- the fully-observed
sub-matrix. The quantity that decides whether a block is usable is the number
of compounds in it, so the question is a search: over all 4-target subsets of
BindingDB, which have the largest fully-observed sub-matrix?

The exact search is infeasible (460 targets clear 400 compounds each, so C(460,4)
is 1.8e9 subsets) and the objective is not submodular in a way that would make
a bound easy, so this greedy-grows from every target in turn: start at a seed,
repeatedly add whichever target leaves the largest intersection. Duplicates are
collapsed, because many seeds converge on the same block. The result is a lower
bound on the best block per family, which is all that is needed -- the output is
a shortlist to run campaigns on, not a claim about optimality.

The check that the search is sane is that JAK (1,563) and CYP (1,472) come out
first and second without being told to. ``FAMILIES`` in
``dense_block_campaign.py`` is this shortlist, labelled.

Sizes printed here are *before* the censoring filter, which removes compounds
sitting on an assay's reporting ceiling (``dense_block_h1.drop_censored``) and
costs anywhere from 42% (JAK) to 77% (CYP) of a block. ``--labelled`` re-reports
the registry's families after that filter, which is the size a campaign sees.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

SOURCE = Path("data/cache/tdc/bindingdb_ic50.csv")
NAME_CACHE = Path("data/cache/uniprot_names.json")


def uniprot_names(accessions):
    """Gene symbols for accessions, so a block can be read rather than decoded.

    Whether a block is "one family or four" is the judgement the whole
    cross-family question turns on, and it cannot be made against a column of
    accessions. Cached, and soft-failing: offline, callers still get their
    counts and see the accessions.
    """
    cache = json.loads(NAME_CACHE.read_text()) if NAME_CACHE.exists() else {}
    missing = sorted(set(accessions) - set(cache))
    for i in range(0, len(missing), 100):
        url = ("https://rest.uniprot.org/uniprotkb/accessions?accessions="
               + ",".join(missing[i:i + 100])
               + "&fields=accession,gene_primary&format=json")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                for e in json.load(r).get("results", []):
                    genes = e.get("genes") or [{}]
                    cache[e["primaryAccession"]] = (
                        genes[0].get("geneName", {}).get("value", "?"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"    (UniProt lookup failed: {exc}; using accessions)")
            break
    NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    NAME_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def incidence(source=SOURCE, *, min_per_target=400):
    """Binary compound x target matrix over targets with enough measurements.

    Read in chunks for the reason ``build_block`` gives: the dump is 839 MB and
    reading it whole costs several GB. Only the pair is kept -- the potency is
    needed to drop non-positive rows and nothing else, because co-measurement
    is a question about which cells are filled, not what is in them.
    """
    if not source.exists():
        raise SystemExit(f"{source} not found; it is the TDC BindingDB IC50 dump.")
    rows = []
    for chunk in pd.read_csv(source, usecols=["X1", "ID2", "Y"], chunksize=500_000):
        chunk = chunk.dropna()
        chunk = chunk[(chunk.Y > 0) & (chunk.Y < 1e9)]
        rows.append(chunk[["X1", "ID2"]])
    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    n_pairs = len(df)
    cnt = df.ID2.value_counts()
    df = df[df.ID2.isin(cnt[cnt >= min_per_target].index)]
    ci, _ = pd.factorize(df.X1)
    ti, targets = pd.factorize(df.ID2)
    M = sparse.csr_matrix((np.ones(len(df), np.int32), (ci, ti)),
                          shape=(ci.max() + 1, len(targets)))
    return M, list(targets), n_pairs


def search(M, targets, *, size=4, min_pair=50):
    """Greedy block growth from every seed, deduplicated and ranked.

    ``min_pair`` prunes the candidate list at each step to targets sharing at
    least that many compounds with the one just added. Without it the inner
    loop is 460 intersections per step per seed; with it the search is seconds.
    It can only lose blocks whose members are pairwise sparse, which are not
    blocks worth running a campaign on.
    """
    C = (M.T @ M).toarray()
    Mc = M.tocsc()
    cols = [np.asarray(Mc[:, j].todense()).ravel().astype(bool)
            for j in range(len(targets))]

    def n_common(idx):
        m = cols[idx[0]].copy()
        for j in idx[1:]:
            m &= cols[j]
        return int(m.sum())

    seen, out = set(), []
    for seed in range(len(targets)):
        cur = [seed]
        while len(cur) < size:
            best, best_n = None, 0
            for j in np.where(C[cur[-1]] >= min_pair)[0]:
                j = int(j)
                if j in cur:
                    continue
                n = n_common(cur + [j])
                if n > best_n:
                    best, best_n = j, n
            if best is None or best_n < min_pair:
                break
            cur.append(best)
        if len(cur) < size:
            continue
        key = tuple(sorted(cur))
        if key in seen:
            continue
        seen.add(key)
        out.append({"n": n_common(list(key)),
                    "uniprot": [targets[j] for j in key]})
    out.sort(key=lambda r: -r["n"])
    return out


MIN_POOL = 160
"""A block smaller than this cannot carry the protocol: the history is
``H x (T-1)`` distinct compounds drawn clear of the top-k, which at H=50 over
three auxiliaries is 150 + 10. Below it the "free history" would cover nearly
the whole library and the experiment stops meaning what it is supposed to."""

def family_properties(pile_frac=0.02, families=None, source=None, potency=None):
    """Per-family block size and task correlation, for the runnable families.

    Spearman rather than Pearson -- the campaign metric is a top-k recall, which
    depends on the ranking and not the scale, and one family (DPP) has a target
    whose potencies span a different range from its relatives'.

    ``families``/``source``/``potency`` default to the registry over the
    BindingDB dump. Pointing them at a kinome panel asks the same question of a
    different library, which is the only way to tell whether "co-measurement is
    family-shaped" is a fact about drug discovery or a fact about BindingDB.
    """
    from dense_block_h1 import build_block, drop_censored, nanomolar_potency

    if families is None:
        from dense_block_campaign import FAMILIES
        families = FAMILIES
    kw = {"potency": potency or nanomolar_potency}
    if source is not None:
        kw["source"] = source

    prop = {}
    for fam, mapping in families.items():
        block, _ = build_block(mapping, **kw)
        tg = [t for t in mapping.values() if t in block.columns]
        clean, _ = drop_censored(block, tg, pile_frac=pile_frac, verbose=False)
        v = clean[tg].to_numpy(float)
        # A block the censoring filter has emptied cannot carry a correlation.
        # scipy returns a bare nan there rather than a matrix, so guard on the
        # shape instead of the row count: pointing this at another library is
        # exactly how a block arrives with two compounds left in it.
        R = np.atleast_2d(spearmanr(v).statistic) if len(v) >= 5 else np.empty((1, 1))
        off = R[np.triu_indices(len(tg), 1)] if R.shape[0] == len(tg) else [np.nan]
        prop[fam] = {
            "targets": tg,
            "n_raw": len(block),
            "n_clean": len(clean),
            "runnable": len(clean) >= MIN_POOL,
            # What share of the library the free history covers at H=50. This
            # is the quantity section [7] shows the gain tracks, and it is also
            # the reason the pool-size test above is not sufficient on its own:
            # cathepsin clears MIN_POOL at 167 compounds, but its history would
            # then be 150 of the 157 compounds outside the top-k, so "free
            # history on relatives" would mean labels on 96% of the library.
            # That is the degenerate end of the coverage axis, not a family the
            # method can be tested on, so it is listed and not run.
            "coverage_at_H50": float(150 / max(len(clean) - 10, 1)),
            "median_rho": float(np.median(off)),
            # Mean correlation of each target against the other three: the
            # support the multi-task arm actually has when that target is the
            # held-out one.
            "aux_rho": {t: float(np.mean(np.delete(R[i], i)))
                        if R.shape[0] == len(tg) else float("nan")
                        for i, t in enumerate(tg)},
        }
    return prop


def print_families(prop) -> None:
    """The one rendering of the family table, shared with the analysis.

    It used to exist twice, and the copies had already diverged: this script
    printed raw/clean/targets while `window_across_families` printed the same
    rows with the task correlation, the history coverage and the too-small
    markers. Two renderings of one table is how a reader ends up trusting
    whichever they happened to run.
    """
    print(f"      {'family':<10}{'raw':>7}{'clean':>7}{'rho':>7}{'cov':>7}  targets")
    for fam, q in sorted(prop.items(), key=lambda kv: -kv[1]["n_clean"]):
        mark = " " if q["runnable"] else "x"
        print(f"    {mark} {fam:<10}{q['n_raw']:>7,}{q['n_clean']:>7,}"
              f"{q['median_rho']:>+7.2f}{min(q['coverage_at_H50'], 9.99):>6.0%}  "
              f"{' '.join(q['targets'])}")
    print(f"    x = fewer than {MIN_POOL} compounds survive the censoring "
          "filter; the protocol cannot run.")
    print("    cov = share of the library the free history covers at H=50; "
          "above ~90% the")
    print("    history is the library and the family is not a usable test "
          "(cathepsin).")


def blocks_as_families(path, top):
    """A scan artefact's top blocks, labelled by gene, as a FAMILIES-shaped dict.

    The registry is a hand-labelled list of blocks; a scan output is the same
    thing unlabelled. Converting one to the other is what lets `--labelled`
    report sizes and task correlations for a panel it has never seen, with no
    second copy of that reporting code.
    """
    blocks = json.loads(Path(path).read_text())["blocks"][:top]
    names = uniprot_names([a for b in blocks for a in b["uniprot"]])
    return {f"k{i:02d}": {a: names.get(a, a) for a in b["uniprot"]}
            for i, b in enumerate(blocks, 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=4, help="targets per block")
    p.add_argument("--source", type=Path, default=SOURCE,
                   help="any dump with ID1,X1,ID2,X2,Y columns; the kinome "
                        "panels from fetch_kinase_panels.py read the same way")
    p.add_argument("--min-per-target", type=int, default=400)
    p.add_argument("--pile-frac", type=float, default=0.02,
                   help="censoring threshold, as in dense_block_campaign")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--labelled", action="store_true",
                   help="skip the search; report the registry's families at "
                        "their post-censoring size, which is what a campaign sees")
    p.add_argument("--blocks-from", metavar="SCAN.json",
                   help="with --labelled, report a previous scan's top blocks "
                        "instead of the registry")
    p.add_argument("--potency", choices=["nanomolar", "identity"],
                   default="nanomolar",
                   help="how the source's Y becomes 'higher is better'. "
                        "nanomolar = pIC50 from IC50/nM (BindingDB); "
                        "identity = already a potency (KIBA score)")
    p.add_argument("--out", default="results/dense_blocks_scan.json")
    args = p.parse_args(argv)

    if args.labelled:
        from dense_block_h1 import nanomolar_potency
        potency = {"nanomolar": nanomolar_potency, "identity": lambda y: y}
        families = (blocks_as_families(args.blocks_from, args.top)
                    if args.blocks_from else None)
        prop = family_properties(args.pile_frac, families, args.source,
                                 potency[args.potency])
        print_families(prop)
        payload = {"labelled": prop, "source": str(args.source)}
    else:
        M, targets, n_pairs = incidence(args.source,
                                        min_per_target=args.min_per_target)
        print(f"{n_pairs:,} unique (compound, target) pairs; "
              f"{len(targets)} targets with >= {args.min_per_target}")
        blocks = search(M, targets, size=args.size)
        for r in blocks[:args.top]:
            print(f"{r['n']:6,}  {' '.join(r['uniprot'])}")
        payload = {"min_per_target": args.min_per_target, "size": args.size,
                   "n_pairs": n_pairs, "n_targets": len(targets),
                   "blocks": blocks[:args.top]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

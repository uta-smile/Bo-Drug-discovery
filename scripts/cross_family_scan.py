"""Real weak relatives, not synthetic ones: the other half of the threshold.

    python scripts/cross_family_scan.py --arms stl
    python scripts/cross_family_scan.py --arms stl mt        # slow

`seed_quality_ladder.py` located the coupling below which seeding stops paying,
but it did so by *degrading an auxiliary uniformly at random*. Real weak
coupling is structured: a relative that ranks one chemotype well and another not
at all. The synthetic ladder therefore brackets the crossing under the most
favourable kind of weakness, and the number it produced cannot be quoted as a
property of weak relatives until a scan over genuinely weak ones agrees with it.
This is that scan.

## Why a new search

`find_dense_blocks.search` grows a block greedily by *maximum shared compounds*.
Kinases that are co-measured are kinases that someone profiled together, and
those are relatives -- which is precisely why all 15 blocks that search returned
sit at task correlation >= 0.277 and why the sweep's zero crossing was an
extrapolation. Searching for size finds families. To reach low coupling the
search objective has to be coupling itself.

## The selection rule, stated so it can be criticised

1. Draw `--samples` random 4-subsets of the targets carrying at least
   `--min-per-target` measurements. Random, not greedy: section 34's standing
   lesson is that effects which look clean on hand-picked families do not
   survive an unbiased draw.
2. Keep those whose fully-observed block has at least `--min-pool` compounds.
3. Compute the median off-diagonal Spearman on the **raw** block -- the matrix
   the campaign actually runs on, not the censoring-filtered one. (Measuring it
   on the filtered block is a mistake this project already made once: on KIBA
   the filter leaves 25-99 of ~900 rows and reports a correlation the campaign
   never sees.)
4. Stratify by that correlation into bins and take `--per-bin` blocks from each,
   choosing within a bin the blocks whose size is **closest to the global
   median** rather than the largest. Size is a nuisance variable here: taking
   the biggest block in each bin would let block size drift with coupling and
   confound the very relation being fitted. The run reports the residual
   correlation between size and coupling so the reader can check.
5. Reject a block sharing more than `--max-overlap` targets with one already
   selected, so the panel is not three views of one family.

This is a selection *on the x axis*, deliberately: it is how you measure a
threshold rather than a population mean. The marginal distribution of coupling
here is therefore not KIBA's own, and no claim about "the typical panel" should
be read off it.

## What agreement would mean

If the crossing found here lands near the ladder's (+0.062 on `stl`), the
threshold is a property of coupling and not of how the coupling was weakened,
and it can be quoted plainly. If it lands higher, structured weakness is worse
than uniform noise at equal correlation, and the ladder's number is an
optimistic bound -- which is the direction one should expect and the reason this
run is owed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import init_headroom  # noqa: E402
from dense_block_campaign import PANELS  # noqa: E402
from find_dense_blocks import uniprot_names  # noqa: E402
from seed_rho_curve import crossing  # noqa: E402

RULES = ("aux_top", "aux_scaffold_nn", "maxmin")
SOURCE = Path("data/cache/tdc/kiba_score.csv")
BINS = (-1.0, 0.00, 0.10, 0.20, 0.28, 0.40, 1.01)
"""Edges chosen so that four of the six bins lie *below* 0.28, the floor of
everything the size-greedy search returned. The point of the run is the region
the existing panel does not cover."""


def panel_matrix(source: Path, *, min_per_target: int, potency="identity",
                 chunksize=500_000):
    """Compound x target value matrix with NaN for unmeasured, plus labels.

    Read in chunks for the reason ``find_dense_blocks.incidence`` gives: the
    BindingDB dump is 839 MB and reading it whole costs several GB.

    ``potency`` must match the direction of the source's ``Y``. ``identity``
    is for a score that is already higher-is-better (KIBA); ``nanomolar`` is
    for a concentration, where lower is better (BindingDB IC50, Davis Kd).
    Getting this wrong inverts the objective, and since Spearman is
    rank-based a *pairwise* correlation survives the inversion unchanged --
    so a wrong setting shows up nowhere in the coupling axis and only in the
    campaign, which is the worst place to find it.
    """
    frames = []
    for chunk in pd.read_csv(source, usecols=["X1", "ID2", "Y"],
                             chunksize=chunksize):
        chunk = chunk.dropna()
        if potency == "nanomolar":
            chunk = chunk[(chunk.Y > 0) & (chunk.Y < 1e9)]
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    # A dump can carry several readings of one (compound, target); keep the
    # median, which is what `build_block` does.
    df = df.groupby(["X1", "ID2"], sort=False, as_index=False).Y.median()
    cnt = df.ID2.value_counts()
    df = df[df.ID2.isin(cnt[cnt >= min_per_target].index)]
    ci, compounds = pd.factorize(df.X1)
    ti, targets = pd.factorize(df.ID2)
    y = df.Y.to_numpy(float)
    if potency == "nanomolar":
        y = 9.0 - np.log10(y)
    # Column store rather than a dense matrix: per target, the sorted
    # compound indices and their values. A dense array is fine for KIBA
    # (2,068 x 173) and is 1.5 GB of mostly NaN for BindingDB
    # (399,237 x 460).
    order = np.lexsort((ci, ti))
    ci, ti, y = ci[order], ti[order], y[order]
    bounds = np.searchsorted(ti, np.arange(len(targets) + 1))
    cols = [(ci[bounds[j]:bounds[j + 1]], y[bounds[j]:bounds[j + 1]])
            for j in range(len(targets))]
    return cols, list(compounds), list(targets)


def common_rows(cols, idx):
    """Compound indices measured on every target in ``idx``."""
    shared = cols[idx[0]][0]
    for j in idx[1:]:
        shared = np.intersect1d(shared, cols[j][0], assume_unique=True)
        if not len(shared):
            break
    return shared


def uncensored(V, pile_frac):
    """Rows touching no value held by more than `pile_frac` of its column.

    The same rule `dense_block_campaign.drop_censored` applies, restated here
    because selection has to see what the campaign will see.
    """
    keep = np.ones(len(V), bool)
    for k in range(V.shape[1]):
        v, c = np.unique(V[:, k], return_counts=True)
        keep &= ~np.isin(V[:, k], v[c / len(V) > pile_frac])
    return keep


def block_values(cols, idx, shared):
    """The fully observed sub-matrix for ``idx`` on ``shared``, as columns."""
    out = np.empty((len(shared), len(idx)))
    for k, j in enumerate(idx):
        ind, val = cols[j]
        out[:, k] = val[np.searchsorted(ind, shared)]
    return out


def block_rho(cols, idx, shared=None, pile_frac=None):
    """Median off-diagonal Spearman among a block's targets.

    On the raw block by default. With ``pile_frac`` the censoring filter runs
    first and both the correlation and the reported size are of the filtered
    block -- which is what the campaign then runs on.

    Returns ``None`` when a column is constant, which makes the correlation
    undefined rather than zero.
    """
    if shared is None:
        shared = common_rows(cols, idx)
    if len(shared) < 3:
        return None, int(len(shared))
    sub = block_values(cols, idx, shared)
    if pile_frac is not None:
        sub = sub[uncensored(sub, pile_frac)]
        shared = shared[:len(sub)]  # only the count is used downstream
        if len(sub) < 3:
            return None, int(len(sub))
    rs = []
    for a, b in itertools.combinations(range(len(idx)), 2):
        if len(np.unique(sub[:, a])) < 3 or len(np.unique(sub[:, b])) < 3:
            return None, int(len(shared))
        rs.append(float(spearmanr(sub[:, a], sub[:, b]).statistic))
    return float(np.median(rs)), int(len(shared))


def co_measurement(cols, n_targets, min_pool):
    """For each target, the targets sharing at least `min_pool` compounds.

    Needed only by the seeded sampler. Computed from a sparse incidence
    product, which is 460 x 460 even when the panel is 399,237 compounds wide.
    """
    from scipy import sparse
    n_comp = max((c[0].max() + 1) for c in cols if len(c[0]))
    rows = np.concatenate([c[0] for c in cols])
    colj = np.concatenate([np.full(len(c[0]), j, dtype=np.int32)
                           for j, c in enumerate(cols)])
    M = sparse.csr_matrix((np.ones(len(rows), np.int32), (rows, colj)),
                          shape=(int(n_comp), n_targets))
    C = (M.T @ M).toarray()
    np.fill_diagonal(C, 0)
    return [np.flatnonzero(C[j] >= min_pool) for j in range(n_targets)]


def candidates(cols, targets, *, samples, min_pool, size, rng, seeded=False,
               pile_frac=None):
    """`size`-subsets that clear `min_pool`, with their coupling.

    ``seeded=False`` draws uniformly and keeps the rng call sequence the
    unseeded KIBA selection was made with. ``seeded=True`` draws one target
    uniformly and the rest from those co-measured with it, which is the only
    way to find blocks at all on a sparse panel: 40,000 uniform draws over
    BindingDB's 460 targets returned zero. It biases the draw toward
    co-measurement, not toward correlation.
    """
    neigh = (co_measurement(cols, len(targets), min_pool) if seeded else None)
    seen, out = set(), []
    for _ in range(samples):
        if seeded:
            seed = int(rng.integers(len(targets)))
            pool = neigh[seed]
            if len(pool) < size - 1:
                continue
            rest = rng.choice(pool, size=size - 1, replace=False)
            idx = tuple(sorted({seed, *rest.tolist()}))
            if len(idx) != size:
                continue
        else:
            idx = tuple(sorted(rng.choice(len(targets), size=size,
                                          replace=False).tolist()))
        if idx in seen:
            continue
        seen.add(idx)
        shared = common_rows(cols, list(idx))
        if len(shared) < min_pool:
            continue
        rho, n_raw = block_rho(cols, list(idx), shared, pile_frac=pile_frac)
        if rho is None or n_raw < min_pool:
            continue
        out.append({"idx": list(idx), "n": n_raw, "rho": rho,
                    "uniprot": [targets[j] for j in idx]})
    return out


def stratify(cands, *, per_bin, max_overlap, bins=BINS):
    """Balanced pick across coupling bins, size held as close to constant as
    the data allows and near-duplicate blocks rejected."""
    if not cands:
        return []
    target_n = float(np.median([c["n"] for c in cands]))
    chosen: list[dict] = []
    for lo, hi in zip(bins, bins[1:]):
        pool = sorted((c for c in cands if lo <= c["rho"] < hi),
                      key=lambda c: abs(c["n"] - target_n))
        taken = 0
        for c in pool:
            if taken >= per_bin:
                break
            if any(len(set(c["idx"]) & set(d["idx"])) > max_overlap
                   for d in chosen):
                continue
            chosen.append({**c, "bin": f"[{lo:.2f}, {hi:.2f})"})
            taken += 1
    chosen.sort(key=lambda c: c["rho"])
    return chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--potency", choices=["identity", "nanomolar"],
                   default=None,
                   help="direction of the source's Y. Defaults to whatever "
                        "PANELS pins to this source, so the pairing cannot be "
                        "forgotten; pass it explicitly for a source the "
                        "registry does not know (the BindingDB dump is "
                        "nanomolar)")
    p.add_argument("--min-per-target", type=int, default=160)
    p.add_argument("--min-pool", type=int, default=300,
                   help="compounds in the fully observed block. The history is "
                        "H x (T-1) distinct compounds drawn clear of the "
                        "top-k, so at H=50 over three auxiliaries a block "
                        "below ~300 has the history covering half the library")
    p.add_argument("--samples", type=int, default=40_000)
    p.add_argument("--pile-frac", type=float, default=None,
                   help="apply the censoring filter, dropping compounds that "
                        "touch any value held by more than this share of a "
                        "column, and select *and run* on the filtered block. "
                        "Off by default: on KIBA the filter leaves 25-99 of "
                        "~900 rows, so it would measure a correlation no "
                        "campaign sees. Required on the BindingDB dump, whose "
                        "co-measured blocks carry reporting ceilings up to 95% "
                        "of a column")
    p.add_argument("--seeded", action="store_true",
                   help="draw one target uniformly and the rest from those "
                        "co-measured with it. Off by default so the stored "
                        "KIBA selection reproduces; required on a sparse "
                        "panel such as the BindingDB dump, where a uniform "
                        "draw finds no blocks at all")
    p.add_argument("--per-bin", type=int, default=3)
    p.add_argument("--max-overlap", type=int, default=2)
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=["stl"])
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--history", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scan-out", default="results/cross_family_blocks.json")
    p.add_argument("--out-dir", default="results/cross_family")
    p.add_argument("--out", default="results/cross_family_scan.json")
    p.add_argument("--dry-run", action="store_true",
                   help="select and report the blocks, run no campaigns")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.potency is None:
        # Resolve from the registry rather than defaulting to a direction. A
        # silent default here is how Davis ended up scored as if higher Kd
        # were better.
        match = [v["potency"] for v in PANELS.values()
                 if Path(v["source"]).as_posix() == args.source.as_posix()]
        if not match:
            p.error(f"{args.source} is not in the PANELS registry, so the "
                    f"direction of its Y cannot be looked up. Pass --potency "
                    f"explicitly: 'identity' if higher Y is better, "
                    f"'nanomolar' if Y is a concentration.")
        args.potency = match[0]

    t0 = time.time()
    cols, compounds, targets = panel_matrix(args.source,
                                            min_per_target=args.min_per_target,
                                            potency=args.potency)
    print(f"{args.source.name}: {len(compounds):,} compounds x {len(targets)} "
          f"targets with >= {args.min_per_target} measurements, "
          f"potency {args.potency} ({time.time() - t0:.0f}s)")
    # A panel can be rich in targets and poor in compounds, and this protocol
    # needs the opposite. Davis is the case in point: a complete 68 x 379
    # matrix, so every target carries exactly 68 measurements and the library
    # is 68 compounds wide -- against a history of H x (T-1) = 150 distinct
    # compounds and a budget of 60. Say so rather than dying inside the
    # sampler.
    V = cols  # name kept for the guard message below
    if len(targets) < args.size:
        raise SystemExit(
            f"only {len(targets)} target(s) carry >= {args.min_per_target} "
            f"measurements, fewer than the {args.size} a block needs.\n"
            f"  This panel has {len(compounds):,} compounds in total. The protocol "
            f"draws H x (T-1) distinct history compounds clear of the top-k "
            f"and then spends a budget on the rest, so a panel narrower than a "
            f"few hundred compounds cannot carry it at any --min-per-target.\n"
            f"  A target-rich, compound-poor panel is not a smaller version of "
            f"this experiment; it is a different one.")

    rng = np.random.default_rng(args.seed)
    cands = candidates(cols, targets, samples=args.samples,
                       min_pool=args.min_pool, size=args.size, rng=rng,
                       seeded=args.seeded, pile_frac=args.pile_frac)
    if not cands:
        raise SystemExit(
            f"no {args.size}-subset of {len(targets)} targets shares "
            f"{args.min_pool} compounds in {args.samples:,} draws.\n"
            f"  On a sparse panel a uniform draw will not find one -- pass "
            f"--seeded to draw the block around a co-measured seed, or lower "
            f"--min-pool.")
    print(f"{len(cands):,} random {args.size}-subsets clear {args.min_pool} "
          f"compounds; coupling spans "
          f"{min(c['rho'] for c in cands):+.3f} to "
          f"{max(c['rho'] for c in cands):+.3f} ({time.time() - t0:.0f}s)")

    chosen = stratify(cands, per_bin=args.per_bin,
                      max_overlap=args.max_overlap)
    names = uniprot_names([a for c in chosen for a in c["uniprot"]])
    # The scan artefact, in the shape `--blocks-from` already reads. Labels are
    # assigned in coupling order, so x01 is the weakest block in the panel.
    scan = {"source": args.source.as_posix(), "potency": args.potency,
            "min_per_target": args.min_per_target, "size": args.size,
            "selection": "random subsets, stratified by raw-block coupling",
            "bins": list(BINS), "per_bin": args.per_bin,
            "min_pool": args.min_pool, "samples": args.samples,
            "seeded": bool(args.seeded), "pile_frac": args.pile_frac,
            "blocks": [{"n": c["n"], "uniprot": c["uniprot"]} for c in chosen]}
    out_scan = Path(args.scan_out)
    # Running a second panel through the default path silently replaced the
    # first panel's blocks, and the report that referenced them kept pointing
    # at the file. Labels are positional (k01..kNN), so a swapped artefact does
    # not error -- it relabels. Refuse instead.
    if out_scan.exists() and not args.force:
        prev = json.loads(out_scan.read_text(encoding="utf-8"))
        # Compare the whole selection signature, not just the source. A
        # different --samples on the *same* panel picks different blocks, and
        # that is just as silent a relabelling as a different panel. It is how
        # this file got overwritten twice while the report referencing it kept
        # pointing at the same path.
        sig = ("source", "potency", "min_per_target", "size", "per_bin",
               "min_pool", "samples")
        diff = [k for k in sig if k in prev and prev.get(k) != scan.get(k)]
        if diff:
            raise SystemExit(
                f"{out_scan} was built with different settings: "
                + ", ".join(f"{k}={prev.get(k)!r} not {scan.get(k)!r}"
                            for k in diff)
                + ".\n  Block labels are positional, so overwriting would "
                "silently relabel every artefact that references this file.\n"
                "  Pass --scan-out with a different path, or --force if you "
                "mean to replace it.")
    out_scan.parent.mkdir(parents=True, exist_ok=True)
    out_scan.write_text(json.dumps(scan, indent=2), encoding="utf-8")

    print(f"\n    {'label':<7}{'rho':>8}{'n':>7}  {'bin':<16}targets")
    for i, c in enumerate(chosen, 1):
        print(f"    {'k%02d' % i:<7}{c['rho']:>+8.3f}{c['n']:>7,}  "
              f"{c['bin']:<16}"
              + " ".join(names.get(a, a) for a in c["uniprot"]))

    # Size must not drift with coupling, or the fit below is confounded.
    if len(chosen) >= 3:
        s = spearmanr([c["rho"] for c in chosen], [c["n"] for c in chosen])
        print(f"\n    size vs coupling among selected blocks: "
              f"rho = {s.statistic:+.3f} (p = {s.pvalue:.3f})"
              + ("  <- CONFOUNDED, read the fit with care"
                 if s.pvalue < 0.05 else "  (independent, as intended)"))
    print(f"    wrote {args.scan_out}")
    if args.dry_run:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(chosen, 1):
        label = f"k{i:02d}"
        for arm in args.arms:
            dest = out_dir / f"{label}_{arm}.json"
            if args.force or not dest.exists():
                argv_ = ["--blocks-from", args.scan_out, "--family", label,
                         "--source", str(args.source),
                         "--potency", args.potency,
                         "--history-excludes-top",
                         "--arm", arm, "--candidates", str(args.candidates),
                         "--repeats", str(args.repeats),
                         "--budget", str(args.budget),
                         "--history", str(args.history),
                         "--out", str(dest)]
                # Selection and campaign must see the same block.
                argv_ += (["--pile-frac", str(args.pile_frac)]
                          if args.pile_frac is not None
                          else ["--keep-censored"])
                if init_headroom.main(argv_) != 0:
                    raise SystemExit(f"{label}/{arm} failed")
            rep = json.loads(dest.read_text(encoding="utf-8"))
            by = {r["label"]: r for r in rep["rows"]}
            rows.append({
                "label": label, "arm": arm, "rho_raw": c["rho"], "n": c["n"],
                "bin": c["bin"], "uniprot": c["uniprot"],
                "targets": [names.get(a, a) for a in c["uniprot"]],
                "rho_eff": rep["rho_eff"],
                "ceiling": by["hindsight ceiling"]["delta_mean"],
                **{r: by[r]["delta_mean"] for r in RULES if r in by},
                **{f"{r}_sem": by[r]["delta_sem"] for r in RULES if r in by},
                "gate_inert": (rep.get("gate") or {}).get("inert"),
                "audit_clean": rep["audit"]["clean"],
                "artefact": dest.as_posix()})
            print(f"  {label} {arm:<4s} rho={c['rho']:+.3f} "
                  f"rho_eff={rep['rho_eff']:+.3f} "
                  + "  ".join(f"{r}={rows[-1].get(r, float('nan')):+.3f}"
                              for r in ("aux_top", "aux_scaffold_nn"))
                  + f"   ({time.time() - t0:.0f}s)", flush=True)

    fits = {}
    for arm in args.arms:
        sel = [r for r in rows if r["arm"] == arm]
        for rule in RULES:
            for xkey in ("rho_raw", "rho_eff"):
                x = [r[xkey] for r in sel if rule in r and r[xkey] is not None]
                y = [r[rule] for r in sel if rule in r and r[xkey] is not None]
                f = crossing(x, y)
                if f:
                    f["x_min"], f["x_max"] = float(min(x)), float(max(x))
                    f["interpolated"] = bool(
                        f["crossing"] is not None
                        and f["x_min"] <= f["crossing"] <= f["x_max"])
                    f["spearman_p"] = float(spearmanr(x, y).pvalue)
                    f["n_blocks"] = len(x)
                    f["mean"] = float(np.mean(y))
                    f["sem"] = float(np.std(y, ddof=1) / np.sqrt(len(y)))
                    f["n_positive"] = int(sum(v > 0 for v in y))
                fits[f"{arm}/{rule}/{xkey}"] = f

    print(f"\n    {'arm/rule/x':<30s}{'rho(x,gain)':>12s}{'p':>8s}"
          f"{'crossing':>11s}{'95% CI':>18s}{'in range':>10s}")
    print("    " + "-" * 89)
    for key, f in fits.items():
        if not f:
            continue
        ci = (f"[{f['ci_lo']:.2f}, {f['ci_hi']:.2f}]"
              if f.get("ci_lo") is not None else "n/a")
        cr = f"{f['crossing']:+.3f}" if f["crossing"] is not None else "n/a"
        print(f"    {key:<30s}{f['spearman_rho_vs_gain']:>+12.3f}"
              f"{f['spearman_p']:>8.4f}{cr:>11s}{ci:>18s}"
              f"{('YES' if f['interpolated'] else 'no'):>10s}")

    # The comparison this script exists to make.
    LADDER = {"stl": 0.062, "mt": 0.013}
    for arm in args.arms:
        f = fits.get(f"{arm}/aux_top/rho_eff")
        if f and f.get("crossing") is not None:
            lo, hi = f.get("ci_lo"), f.get("ci_hi")
            agree = (lo is not None and lo <= LADDER[arm] <= hi)
            print(f"\n    {arm}: cross-family crossing {f['crossing']:+.3f}, "
                  f"synthetic ladder {LADDER[arm]:+.3f} -> "
                  + ("the ladder's number is inside this CI; structured and "
                     "uniform weakness agree"
                     if agree else
                     "the ladder's number is OUTSIDE this CI; structured "
                     "weakness differs and the ladder is a bound, not the "
                     "answer"))
        low = [r for r in rows if r["arm"] == arm and r["rho_raw"] < 0.28]
        if low:
            pos = sum(r.get("aux_top", 0) > 0 for r in low)
            print(f"    {arm}: on the {len(low)} blocks below rho = 0.28 "
                  f"(which the size-greedy scan never reached), aux_top is "
                  f"positive on {pos}")

    clean = all(r["audit_clean"] for r in rows) if rows else True
    print(f"\n    {len(rows)} runs, audit clean on every one: {clean}")
    report = {"source": args.source.as_posix(), "potency": args.potency,
              "pile_frac": args.pile_frac, "seeded": bool(args.seeded),
              "scan": args.scan_out,
              "selection": scan["selection"], "bins": list(BINS),
              "per_bin": args.per_bin, "min_pool": args.min_pool,
              "samples": args.samples, "n_candidates": len(cands),
              "arms": args.arms, "candidates": args.candidates,
              "repeats": args.repeats, "budget": args.budget,
              "history": args.history, "blocks": chosen, "rows": rows,
              "fits": fits, "ladder_crossing": LADDER,
              "audit_clean": bool(clean)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

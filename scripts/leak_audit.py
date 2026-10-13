"""Which stored results are exposed to the section-40 design leak, and how far?

    python scripts/leak_audit.py
    python scripts/leak_audit.py --json results/leak_audit.json

FINDINGS 40 found that under ``pooled`` (``r = 1`` forced) every history
compound is treated as already observed, so its expected improvement is near
zero and the arm systematically *avoids* the history. ``--history-excludes-top``
guarantees the primary's top-k is outside the history, so avoiding the history
means walking toward the answer. Removing the flag flips the sign.

The size of that leak is set by **how much of the library the history covers**.
Measured directly, as the swing in ``pooled``'s pure-noise gain when the flag is
removed:

===========  ==========  =========
coverage     swing       block
===========  ==========  =========
17%          8.1 points  jak
45%          29.1        cyp
90%          46.0        cyp
===========  ==========  =========

This script does not re-run anything. It reads the stored artefacts, recovers
each one's coverage from its own provenance, and reports which results carry a
``pooled`` arm at what exposure. The interpolation is deliberately crude --
three points, two blocks -- so the number it prints is an order of magnitude and
is labelled as one. The decision it supports is binary: which cells have to be
re-run with the flag crossed before they can be quoted.

``mt`` is reported alongside but is *not* flagged. On both blocks tested its
gain is stable across the flag (JAK +7.66 / +8.28, CYP +9.69 / +11.25), which is
the result section 40 rests on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (coverage, swing in recall points) from FINDINGS 40, `pooled` under a pure
# noise auxiliary. Two blocks, three points; used only to order the exposure.
ANCHORS = ((0.17, 8.05), (0.45, 29.06), (0.90, 46.02))


def coverage(d: dict) -> float | None:
    """Fraction of the campaign library the free history occupies.

    ``history`` is per auxiliary target and there are ``T - 1`` of them, so the
    history is ``history * (T - 1)`` distinct compounds out of ``n_pool`` -- the
    censoring-filtered library the campaign actually runs on, not the raw block.
    """
    prov = d.get("provenance") or {}
    targets = prov.get("targets")
    if not targets and isinstance(d.get("per_target"), dict):
        # Older artefacts predate the provenance block but still name every
        # target they held out, one key each, which is the same count.
        targets = list(d["per_target"])
    n_pool = d.get("n_pool") or prov.get("n_pool")
    h = d.get("history")
    if not targets or not n_pool or h is None:
        return None
    return float(h) * (len(targets) - 1) / float(n_pool)


def expected_swing(cov: float) -> float:
    """Linear interpolation between the measured anchors, clamped at the ends.

    Crude on purpose. Three points cannot support a functional form, and the
    only thing this has to do is sort cells into "ignorable" and "re-run".
    """
    xs = [a for a, _ in ANCHORS]
    ys = [b for _, b in ANCHORS]
    if cov <= xs[0]:
        return ys[0] * cov / xs[0]
    if cov >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= cov <= xs[i + 1]:
            t = (cov - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def scan(root: Path):
    rows = []
    for f in sorted(root.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "arms" not in d:
            continue
        arms = d.get("arms") or {}
        if "pooled" not in arms:
            continue
        cov = coverage(d)
        stl = arms.get("stl", {}).get("recall_mean")
        pld = arms.get("pooled", {}).get("recall_mean")
        mt = arms.get("mt", {}).get("recall_mean")
        rows.append({
            "file": f.name,
            "family": d.get("family"),
            "excludes_top": bool(d.get("history_excludes_top")),
            "history": d.get("history"),
            "n_pool": d.get("n_pool"),
            "coverage": cov,
            "expected_swing": expected_swing(cov) if cov is not None else None,
            "pooled_delta": (None if stl is None or pld is None
                             else (pld - stl) * 100),
            "mt_delta": (None if stl is None or mt is None
                         else (mt - stl) * 100),
        })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--json", type=Path, default=None,
                   help="also write the full table here")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="expected swing, in recall points, above which a cell "
                        "has to be re-run with the flag crossed before its "
                        "pooled number can be quoted")
    args = p.parse_args(argv)

    rows = scan(args.results)
    exposed = [r for r in rows
               if r["excludes_top"] and r["coverage"] is not None
               and r["expected_swing"] >= args.threshold]
    unknown = [r for r in rows if r["coverage"] is None]

    print(f"{len(rows)} stored artefacts carry a `pooled` arm.")
    print(f"{sum(r['excludes_top'] for r in rows)} of them ran with "
          f"--history-excludes-top.\n")

    if exposed:
        print(f"EXPOSED -- expected swing >= {args.threshold:g} recall points. "
              f"The `pooled` number in each of these is not quotable until the "
              f"same cell is run without the flag.\n")
        print(f"  {'file':<42}{'H':>4}{'pool':>7}{'cover':>8}"
              f"{'~swing':>8}{'pooled d':>10}{'mt d':>8}")
        print("  " + "-" * 87)
        for r in sorted(exposed, key=lambda r: -r["expected_swing"]):
            pd = ("--" if r["pooled_delta"] is None
                  else f"{r['pooled_delta']:+.1f}")
            md = "--" if r["mt_delta"] is None else f"{r['mt_delta']:+.1f}"
            print(f"  {r['file']:<42}{r['history']:>4}{r['n_pool']:>7}"
                  f"{r['coverage']*100:>7.0f}%{r['expected_swing']:>8.1f}"
                  f"{pd:>10}{md:>8}")
    else:
        print(f"No artefact exceeds the {args.threshold:g}-point threshold.")

    if unknown:
        print(f"\n{len(unknown)} artefact(s) carry no target list or n_pool, so "
              f"coverage could not be recovered:")
        for r in unknown[:10]:
            print(f"  {r['file']}")

    by_fam = {}
    for r in rows:
        if r["coverage"] is not None:
            by_fam.setdefault(r["family"], []).append(r["coverage"])
    print("\n  coverage by family (min..max over stored runs):")
    for fam, cs in sorted(by_fam.items(), key=lambda kv: -max(kv[1])):
        print(f"    {str(fam):<22}{min(cs)*100:>6.0f}% .. {max(cs)*100:>5.0f}%"
              f"   ({len(cs)} runs)")

    print("\n  `mt` is listed but not flagged: section 40 measured it stable "
          "across\n  the flag on both blocks (JAK +7.66/+8.28, CYP "
          "+9.69/+11.25).")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"threshold": args.threshold, "anchors": ANCHORS,
             "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

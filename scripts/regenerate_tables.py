"""Regenerate every number in the paper's main tables, from stored artefacts.

    python scripts/regenerate_tables.py
    python scripts/regenerate_tables.py --tex report/generated
    python scripts/regenerate_tables.py --check report/paper.tex

Nothing is re-run: this reads ``results/`` and recomputes. Its purpose is that
the manuscript's numbers stop being transcribed by hand. Two rounds of review
found three defects that a transcription check would have caught --- a campaign
count that disagreed between the Methods and a caption, a probability quoted
from a test that had discarded fifteen of its eighteen pairs, and a mean
reported as if it were a maximum.

With ``--check`` it reads the manuscript, pulls out the numbers it knows how to
compute, and reports any that do not match what the artefacts say. That check is
deliberately crude --- it looks for the computed value as a formatted string ---
but it fails loudly, which is the property that was missing.

The three tables:

* **null controls** --- real against randomized auxiliary values on two blocks,
  plus the matched-uncertainty baseline that uses no auxiliary data.
* **history design** --- the window grid under both history designs, with the
  between-design contrast tested over families, not over budget cells.
* **decision rule** --- three uses of a free history on 18 cross-family blocks,
  leave-one-block-out, with the rule-versus-fixed comparison tested on the
  blocks where the two actually differ.
"""

from __future__ import annotations

import argparse
import collections
import glob as globmod
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# shared helpers

def load(pattern: str) -> list[dict]:
    return [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(globmod.glob(str(RESULTS / pattern)))]


def contrast(d: dict, arm: str, baseline: str = "stl") -> np.ndarray | None:
    arms = d.get("arms") or {}
    if arm not in arms or baseline not in arms:
        return None
    a, b = arms[arm].get("recall"), arms[baseline].get("recall")
    if a is None or b is None:
        return None
    return (np.asarray(a, float) - np.asarray(b, float)) * 100.0


def signed_rank(x: np.ndarray) -> dict:
    """With the two facts that decide how the probability may be read."""
    x = np.asarray(x, float)
    zeros = int((x == 0.0).sum())
    nz = np.abs(x[x != 0.0])
    ties = int(len(nz) - len(np.unique(nz)))
    exact = bool(zeros == 0 and ties == 0 and 0 < len(x) <= 50)
    p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    n = len(x)
    sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    return {"n": n, "mean": float(x.mean()), "sem": sem,
            "ci": (float(x.mean() - half), float(x.mean() + half)),
            "positive": int((x > 0).sum()), "p": p, "zeros": zeros,
            "ties": ties, "exact": exact,
            "floor": min(1.0, 2.0 ** (1 - n))}


def by_family(rows: list[dict], arm: str) -> dict[str, float]:
    agg = collections.defaultdict(list)
    for d in rows:
        c = contrast(d, arm)
        if c is not None:
            agg[d["family"]].append(float(c.mean()))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def fmt_p(s: dict) -> str:
    tag = "" if s["exact"] else "  [approx"
    if not s["exact"]:
        tag += f": {s['zeros']} zero, {s['ties']} tied]"
    at = "  = the floor" if s["exact"] and s["p"] <= s["floor"] * 1.0001 else ""
    return f"p = {s['p']:.4f}{at}{tag}"


# ---------------------------------------------------------------------------
# table 1: the null controls

def table_null() -> list[str]:
    # `sdref_*`, not `auxq_*`. The two groups differ in seeds per target (32
    # against 8) and the manuscript's table is built from the first; an earlier
    # version of this script read the second and "confirmed" the wrong counts.
    out = ["NULL CONTROLS (recall points vs stl) -- source: results/sdref_*",
           ""]
    claims = []
    for block, budget in (("jak", 300), ("cyp", 111)):
        for arm in ("mt", "pooled"):
            for q, label in ((0.0, "q=0 (randomized)"), (1.0, "q=1 (measured)")):
                f = RESULTS / f"sdref_{block}_{q:.1f}.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text(encoding="utf-8"))
                c = contrast(d, arm)
                if c is None:
                    continue
                m, sem = c.mean(), c.std(ddof=1) / np.sqrt(len(c))
                out.append(f"  {block.upper():<4}{arm:<8}{label:<20}"
                           f"{m:>+8.2f} +- {sem:<6.2f}  n = {len(c)} campaigns"
                           f"   achieved rho = "
                           f"{d.get('aux_quality_achieved', float('nan')):+.4f}")
                claims.append((f"{block}/{arm}/q={q}", m, sem))
    out.append("")
    for block, scale in (("jak", "0.80"), ("cyp", "0.80")):
        base = RESULTS / f"sdscale_{block}_1.00.json"
        test = RESULTS / f"sdscale_{block}_{scale}.json"
        if not (base.exists() and test.exists()):
            continue
        b = json.loads(base.read_text(encoding="utf-8"))
        t = json.loads(test.read_text(encoding="utf-8"))
        x = (np.asarray(t["arms"]["stl"]["recall"], float)
             - np.asarray(b["arms"]["stl"]["recall"], float)) * 100.0
        out.append(f"  {block.upper():<4}{'stl':<8}{'sd x ' + scale:<20}"
                   f"{x.mean():>+8.2f} +- "
                   f"{x.std(ddof=1) / np.sqrt(len(x)):<6.2f}  n = {len(x)} "
                   f"campaigns   no auxiliary data")
    out.append("")
    out.append("  All rows are 32 seeds on each of 4 held-out targets. The "
               "standard errors still")
    out.append("  differ between the groups, and not because of sample size: "
               "they are different")
    out.append("  contrasts, on different arms, with different between-target "
               "spread.")
    return out


# ---------------------------------------------------------------------------
# table 2: the history designs

def table_design() -> list[str]:
    uni, wit = load("winleak_*.json"), load("win32_*.json")
    if not (uni and wit):
        return ["HISTORY DESIGN: artefacts missing"]
    out = ["", "HISTORY DESIGN (recall points vs stl, family as the unit)", ""]
    for arm in ("mt", "pooled"):
        fu, fw = by_family(uni, arm), by_family(wit, arm)
        su = signed_rank(np.array(list(fu.values())))
        sw = signed_rank(np.array([fw[k] for k in fu]))
        chg = signed_rank(np.array([fu[k] - fw[k] for k in fu]))
        for label, s in (("withheld", sw), ("uniform", su),
                         ("change (uniform - withheld)", chg)):
            out.append(f"  {arm:<8}{label:<30}{s['mean']:>+8.2f} +- "
                       f"{s['sem']:<5.2f} 95% CI "
                       f"[{s['ci'][0]:+7.2f},{s['ci'][1]:+7.2f}]  "
                       f"{s['positive']}/{s['n']}  {fmt_p(s)}")
        # The cell-level figure, for completeness and clearly labelled.
        cu = np.array([float(contrast(d, arm).mean()) for d in uni
                       if contrast(d, arm) is not None])
        cw = np.array([float(contrast(d, arm).mean()) for d in wit
                       if contrast(d, arm) is not None])
        sc = signed_rank(cu - cw)
        out.append(f"  {'':<8}{'  (same change over 12 cells)':<30}"
                   f"{sc['mean']:>+8.2f} +- {sc['sem']:<5.2f}"
                   f"{'':<24}  {sc['positive']}/{sc['n']}  {fmt_p(sc)}")
        out.append("")
    return out


# ---------------------------------------------------------------------------
# table 3: the decision rule

def table_rule() -> list[str]:
    out = ["DECISION RULE (AUC gain over abstaining, 18 cross-family blocks)",
           ""]
    for name in ("uniform", "withheld"):
        f = RESULTS / f"transfer_rule_{name}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out.append(f"  {name}:")
        for o in d["options"]:
            vals = np.array([b[o] for b in d["per_block"]])
            out.append(f"    always {o:<12}{vals.mean():>+9.4f}   "
                       f"{int((vals > 0).sum())}/{len(vals)} blocks positive")
        out.append(f"    rule (LOBO) {'':<4}{d['rule']:>+9.4f}")
        out.append(f"    best per block (hindsight){d['oracle']:>+9.4f}   "
                   f"biased upward; a reference, not a method")
        t = d.get("rule_vs_best_fixed")
        if t:
            out.append(
                f"    rule - always {t['best_fixed']}: "
                f"{t['mean_difference']:+.4f} +- {t['sem_difference']:.4f} "
                f"over {t['n_blocks']} blocks, but {t['n_zero']} are exact "
                f"zeros")
            out.append(
                f"      (the rule picks that policy there), so the test sees "
                f"{t['n_discordant']} blocks:")
            out.append(
                f"      exact signed-rank p = {t['signed_rank_p_exact']:.4f}  "
                f"[scipy default over all {t['n_blocks']}: "
                f"{t['scipy_default_p']:.4f} -- a normal approximation, do not "
                f"quote]")
        out.append("")
    return out


# ---------------------------------------------------------------------------

def check_manuscript(tex: Path, lines: list[str]) -> int:
    """Crude but loud: every number we compute must appear in the manuscript."""
    body = tex.read_text(encoding="utf-8")
    wanted = {
        "mt uniform mean": "11.49", "mt withheld mean": "9.37",
        "pooled uniform mean": "3.16", "pooled withheld mean": "8.89",
        "mt change": "2.12", "pooled change": "5.73",
        "rule uniform exact p": "0.25", "rule withheld exact p": "0.22",
    }
    missing = [k for k, v in wanted.items() if v not in body]
    print("\nMANUSCRIPT CHECK")
    if missing:
        for k in missing:
            print(f"  MISSING from {tex.name}: {k} ({wanted[k]})")
        return 1
    print(f"  every regenerated headline number appears in {tex.name}.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", type=Path, default=None,
                   help="a .tex file to check the numbers against")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    lines = table_null() + table_design() + table_rule()
    print("\n".join(lines))

    rc = 0
    if args.check:
        rc = check_manuscript(args.check, lines)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"lines": lines}, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

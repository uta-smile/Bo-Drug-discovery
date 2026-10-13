"""Structured summaries of every reported result, and the LaTeX generated from them.

    python scripts/summaries.py --json results/summaries.json \
        --tex-dir report/generated

    saved campaign outcomes  ->  structured summaries  ->  LaTeX tables + macros

The previous check searched the manuscript for a formatted number. That passes
even when the right number sits in the wrong row, and it cannot catch a stale
table at all. This replaces it: every table and every headline number in the
abstract is generated from one list of summary records, so changing an
underlying result updates the table and the headline together or not at all.

A summary record carries everything needed to read the number without the
surrounding prose:

    experiment        which artefact group it came from
    contrast          the model comparison, e.g. "mt - stl"
    history           "uniform" or "withheld"
    budget            the assay budget, or a range
    metric            what is measured, e.g. "top-10 recall (points)"
    unit              the aggregation unit: campaign, configuration, family, block
    n                 how many units
    estimate, sem, ci the value and its uncertainty
    test              which test, its p, and whether it was exact
    source            the artefact glob it was computed from

Rounding happens only at render time. The macros are written as
``\newcommand`` so that a headline in the abstract and the cell in the table
are the same number by construction.
"""

from __future__ import annotations

import argparse
import collections
import glob as globmod
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# the record

def record(*, experiment, contrast, history, budget, metric, unit, values,
           source, test="wilcoxon signed-rank, two-sided", name=None):
    """One summary. ``values`` is the per-unit vector the estimate is over.

    ``p`` is computed from the test ``test`` names. Until the October audit it
    was always the signed-rank p, including on the sixteen within-block records
    that name a paired t -- a label that disagreed with its own number. Both
    are kept, since the two answer the same question differently and the
    evidence package carries the record rather than a rendered string.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    zeros = int((x == 0.0).sum())
    nz = np.abs(x[x != 0.0])
    ties = int(len(nz) - len(np.unique(nz)))
    exact = bool(zeros == 0 and ties == 0 and 0 < n <= 50)
    # Ties and zeros put the exact signed-rank distribution out of reach, and
    # the conventions for what to do instead disagree in the third decimal --
    # enough to move a p across 0.05, as the one tie in `mtDesignChangeCell`
    # does. The rule is named here rather than left to a library default:
    # exact when it is available, otherwise the normal approximation with a
    # continuity correction, which is the conservative of the two.
    if not np.any(x != 0):
        p_rank = 1.0
    elif exact:
        p_rank = float(wilcoxon(x, method="exact").pvalue)
    else:
        p_rank = float(wilcoxon(x, method="approx", correction=True).pvalue)
    p_t = float(ttest_1samp(x, 0.0).pvalue) if n > 1 else 1.0
    p = p_t if test.startswith("paired t-test") else p_rank
    exact = exact and not test.startswith("paired t-test")
    return {
        "name": name, "experiment": experiment, "contrast": contrast,
        "history": history, "budget": budget, "metric": metric,
        "unit": unit, "n": n,
        "estimate": float(x.mean()), "sem": sem,
        "ci95": [float(x.mean() - half), float(x.mean() + half)],
        "positive": int((x > 0).sum()),
        "test": test, "p": p, "p_exact": exact,
        "p_signed_rank": p_rank, "p_ttest": p_t,
        "zeros": zeros, "ties": ties,
        "p_floor": min(1.0, 2.0 ** (1 - n)),
        "source": source,
    }


# ---------------------------------------------------------------------------
# readers

def load(pattern):
    return [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(globmod.glob(str(RESULTS / pattern)))]


def contrast_vec(d, arm, baseline="stl"):
    arms = d.get("arms") or {}
    if arm not in arms or baseline not in arms:
        return None
    a, b = arms[arm].get("recall"), arms[baseline].get("recall")
    if a is None or b is None:
        return None
    return (np.asarray(a, float) - np.asarray(b, float)) * 100.0


def per_family(rows, arm):
    agg = collections.defaultdict(list)
    for d in rows:
        v = contrast_vec(d, arm)
        if v is not None:
            agg[d["family"]].append(float(v.mean()))
    return {k: float(np.mean(v)) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# the summaries

def build() -> list[dict]:
    out = []
    uni, wit = load("winleak_*.json"), load("win32_*.json")

    for arm in ("mt", "pooled"):
        fu, fw = per_family(uni, arm), per_family(wit, arm)
        if not fu or not fw:
            continue
        keys = sorted(fu)
        for label, vec, src in (
                ("uniform", [fu[k] for k in keys], "results/winleak_*.json"),
                ("withheld", [fw[k] for k in keys], "results/win32_*.json")):
            out.append(record(
                name=f"{arm}{label.capitalize()}",
                experiment="window grid", contrast=f"{arm} - stl",
                history=label, budget="55-300", metric="top-10 recall (points)",
                unit="protein family", values=vec, source=src))
        out.append(record(
            name=f"{arm}DesignChange",
            experiment="window grid", contrast=f"{arm} - stl, uniform - withheld",
            history="both", budget="55-300", metric="top-10 recall (points)",
            unit="protein family", values=[fu[k] - fw[k] for k in keys],
            source="results/winleak_*.json vs results/win32_*.json"))
        # The same contrast one level down, reported only as a secondary
        # figure: two budgets on one family are not independent.
        cu = [float(contrast_vec(d, arm).mean()) for d in uni
              if contrast_vec(d, arm) is not None]
        cw = [float(contrast_vec(d, arm).mean()) for d in wit
              if contrast_vec(d, arm) is not None]
        out.append(record(
            name=f"{arm}DesignChangeCell",
            experiment="window grid", contrast=f"{arm} - stl, uniform - withheld",
            history="both", budget="55-300", metric="top-10 recall (points)",
            unit="configuration", values=np.array(cu) - np.array(cw),
            source="results/winleak_*.json vs results/win32_*.json"))

    # Null controls, from sdref (not auxq -- different experiment, 8 seeds).
    for block, budget in (("jak", 300), ("cyp", 111)):
        for q, qlabel in ((0.0, "randomized"), (1.0, "measured")):
            f = RESULTS / f"sdref_{block}_{q:.1f}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for arm in ("mt", "pooled"):
                v = contrast_vec(d, arm)
                if v is None:
                    continue
                out.append(record(
                    name=f"null{block.capitalize()}{arm.capitalize()}"
                         f"{'Rand' if q == 0 else 'Real'}",
                    experiment="null control", contrast=f"{arm} - stl",
                    history="withheld", budget=budget,
                    metric="top-10 recall (points)", unit="campaign",
                    values=v, source=f"results/sdref_{block}_{q:.1f}.json",
                    test="paired t-test within block"))

    # The mechanism decomposition. `mt_r0` is the joint model with the task
    # covariance forced to zero and the shared hyperparameters still refitted,
    # so any gain it shows cannot be cross-target transfer.
    for block in ("jak", "cyp"):
        f = RESULTS / f"mech_{block}_q0.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for arm in ("mt_r0", "mt", "pooled"):
            v = contrast_vec(d, arm)
            if v is None:
                continue
            out.append(record(
                name=f"mech{block.capitalize()}"
                     # "MtRZero", not "MtR0": a digit cannot appear in a LaTeX
                     # control sequence, so \valMechJakMtR0 parses as
                     # \valMechJakMtR followed by a stray 0 and \newcommand
                     # fails. Macro names must stay alphabetic.
                     + {"mt_r0": "MtRZero", "mt": "Mt", "pooled": "Pooled"}[arm],
                experiment="mechanism decomposition",
                contrast=f"{arm} - stl, randomized auxiliary",
                history="withheld", budget=d.get("budget"),
                metric="top-10 recall (points)", unit="campaign", values=v,
                source=f"results/mech_{block}_q0.json",
                test="paired t-test within block"))

    # The scale-matched baseline, as a difference against its own c = 1 run.
    for block, scale in (("jak", "0.80"), ("cyp", "0.80")):
        base = RESULTS / f"sdscale_{block}_1.00.json"
        test = RESULTS / f"sdscale_{block}_{scale}.json"
        if not (base.exists() and test.exists()):
            continue
        b = json.loads(base.read_text(encoding="utf-8"))
        t = json.loads(test.read_text(encoding="utf-8"))
        v = (np.asarray(t["arms"]["stl"]["recall"], float)
             - np.asarray(b["arms"]["stl"]["recall"], float)) * 100.0
        out.append(record(
            name=f"scaleMatched{block.capitalize()}",
            experiment="scale-matched baseline",
            contrast=f"stl at c={scale} - stl at c=1",
            history="withheld", budget=t.get("budget"),
            metric="top-10 recall (points)", unit="campaign", values=v,
            source=f"results/sdscale_{block}_{{1.00,{scale}}}.json",
            test="paired t-test within block"))

    return out


# ---------------------------------------------------------------------------
# rendering

def fmt(x, places=2, signed=True):
    s = f"{x:+.{places}f}" if signed else f"{x:.{places}f}"
    return f"${s}$"


def macros(summaries) -> str:
    """One \\newcommand per named summary, so the abstract and the table
    cannot drift apart."""
    lines = ["% Generated by scripts/summaries.py -- do not edit.",
             "% Each macro is rendered from the same summary record that",
             "% produces the corresponding table cell.", ""]
    for s in summaries:
        if not s["name"]:
            continue
        base = s["name"][0].upper() + s["name"][1:]
        lines.append(f"\\newcommand{{\\val{base}}}{{{fmt(s['estimate'])}}}")
        lines.append(f"\\newcommand{{\\sem{base}}}{{${s['sem']:.2f}$}}")
        lines.append(
            f"\\newcommand{{\\ci{base}}}{{$[{s['ci95'][0]:+.2f},"
            f"{s['ci95'][1]:+.2f}]$}}")
        lines.append(f"\\newcommand{{\\n{base}}}{{{s['n']}}}")
        lines.append(f"\\newcommand{{\\pos{base}}}"
                     f"{{{s['positive']}/{s['n']}}}")
        lines.append(f"\\newcommand{{\\p{base}}}{{${s['p']:.3f}$}}")
        lines.append("")
    return "\n".join(lines) + "\n"


def table_design(summaries) -> str:
    by = {s["name"]: s for s in summaries if s["name"]}
    need = ["mtWithheld", "mtUniform", "pooledWithheld", "pooledUniform",
            "mtDesignChange", "pooledDesignChange"]
    if any(k not in by for k in need):
        return ""
    rows = [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Arm & History & Mean & SEM & 95\% CI & Families $>0$ & $p$ \\",
            r"\midrule"]
    for arm in ("mt", "pooled"):
        for hist in ("Withheld", "Uniform"):
            s = by[f"{arm}{hist}"]
            rows.append(
                f"\\{arm}{{}} & {hist.lower()} & {fmt(s['estimate'])} & "
                f"${s['sem']:.2f}$ & "
                f"$[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]$ & "
                f"{s['positive']}/{s['n']} & ${s['p']:.3f}$ \\\\")
        s = by[f"{arm}DesignChange"]
        rows.append(
            f" & \\emph{{change}} & {fmt(s['estimate'])} & ${s['sem']:.2f}$ & "
            f"$[{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}]$ & "
            f"{s['positive']}/{s['n']} & ${s['p']:.3f}$ \\\\")
        if arm == "mt":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def check(tex: Path, summaries) -> int:
    """Every generated macro the manuscript uses must be defined, and no value
    a macro already carries may also appear typed by hand.

    The second half is the one that matters. A text search for "11.49" passes
    whether or not that number is in the right row, and keeps passing after the
    underlying result changes. Finding "11.49" typed into the source means
    somebody copied a number that a macro already renders, which is exactly how
    a table and an abstract drift apart.
    """
    import re
    raw = tex.read_text(encoding="utf-8").split("\n")

    # A per-cell table legitimately contains the same value as a summary -- the
    # null table's measured-panel entry IS one cell of the window grid. Those
    # are not drift, so a region may be exempted, with the reason recorded in
    # the source next to it:
    #
    #     % summaries-exempt: <why this region may hold literal values>
    #     ...
    #     % end-summaries-exempt
    #
    # The exemption is explicit and reviewable. Silently loosening the pattern
    # until it stopped complaining would defeat the check.
    kept, exempt, regions = [], False, []
    for line in raw:
        stripped = line.lstrip()
        if stripped.startswith("% summaries-exempt:"):
            exempt = True
            regions.append(stripped.split(":", 1)[1].strip())
            continue
        if stripped.startswith("% end-summaries-exempt"):
            exempt = False
            continue
        if stripped.startswith("%"):
            continue
        kept.append("" if exempt else line)
    body = "\n".join(kept)

    defined = set()
    for s in summaries:
        if not s["name"]:
            continue
        base = s["name"][0].upper() + s["name"][1:]
        defined |= {f"val{base}", f"sem{base}", f"ci{base}", f"n{base}",
                    f"pos{base}", f"p{base}"}

    used = {m for m in re.findall(r"\\((?:val|sem|ci|pos)[A-Z]\w*)", body)}
    missing = sorted(used - defined)

    hand_typed = []
    for s in summaries:
        if not s["name"]:
            continue
        for value in (f"{s['estimate']:.2f}", f"{abs(s['estimate']):.2f}"):
            if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", body):
                hand_typed.append((s["name"], value))
                break

    print(f"\nMACRO CHECK against {tex.name}")
    print(f"  {len(used)} generated macros used, {len(defined)} defined")
    for why in regions:
        print(f"  exempt region: {why}")
    for m in missing:
        print(f"  UNDEFINED macro used: \\{m}")
    for name, value in hand_typed:
        print(f"  HAND-TYPED {value} appears in the source, but \\val"
              f"{name[0].upper() + name[1:]} already renders it")
    if not missing and not hand_typed:
        print("  every reported value renders from a summary record; "
              "none is typed by hand.")
    return 1 if (missing or hand_typed) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--tex-dir", type=Path, default=None)
    p.add_argument("--check", type=Path, default=None,
                   help="a .tex file: verify every generated macro it uses is "
                        "defined, and that it quotes no hand-typed value that "
                        "a macro already carries")
    args = p.parse_args(argv)

    summaries = build()
    if not summaries:
        raise SystemExit("no artefacts found; nothing to summarise.")

    print(f"{len(summaries)} summary records\n")
    print(f"  {'name':<24}{'unit':<16}{'n':>4}{'estimate':>11}{'sem':>8}"
          f"{'p':>9}{'exact':>7}")
    print("  " + "-" * 80)
    for s in summaries:
        print(f"  {str(s['name']):<24}{s['unit']:<16}{s['n']:>4}"
              f"{s['estimate']:>+11.3f}{s['sem']:>8.3f}{s['p']:>9.4f}"
              f"{'yes' if s['p_exact'] else 'no':>7}")

    if args.tex_dir:
        args.tex_dir.mkdir(parents=True, exist_ok=True)
        (args.tex_dir / "macros.tex").write_text(macros(summaries),
                                                 encoding="utf-8")
        t = table_design(summaries)
        if t:
            (args.tex_dir / "table_design.tex").write_text(t, encoding="utf-8")
        print(f"\nwrote {args.tex_dir}/macros.tex"
              + (f" and table_design.tex" if t else ""))

    rc = 0
    if args.check:
        rc = check(args.check, summaries)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

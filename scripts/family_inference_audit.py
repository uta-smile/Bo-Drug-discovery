"""Recompute every cross-family statistic in the paper, from the artefacts up.

    python scripts/family_inference_audit.py
    python scripts/family_inference_audit.py --out results/family_inference_audit.json

``scripts/summaries.py`` already generates the manuscript's family-level
macros, so a second script that calls it would check nothing. This one reads
``results/`` directly, aggregates with its own code, and compares the result
against the *rendered macros* in ``report/generated/macros.tex`` -- against the
strings the manuscript actually compiles, rather than against the dictionary
that produced them.

Three things are checked, because three different mistakes are possible.

**The unit.** Each cell of the window grid holds 128 campaigns (4 held-out
targets x 32 repeats). Those campaigns share a library, a history draw and an
initial design, so they are not 128 independent observations of "does coupling
help on a new protein family". Every cross-family claim is therefore stated
over six families, and the twelve budget cells are reported only as a
secondary figure. The audit prints the unit and the n beside every quantity
and fails if a family-level macro is not an n=6 statistic.

**The interval and the test.** Recomputed here from the family-level vector:
Student-t 95% interval on n-1 degrees of freedom, and a two-sided Wilcoxon
signed-rank test, exact whenever there are no zeros and no ties. At n=6 the
smallest attainable two-sided p is 2^-5 = 0.031, reached exactly when all six
families agree in sign, so that value carries no more information than the
sign count beside it; the floor is printed with every p.

**The pairing.** Each family-level difference pairs a cell against the same
cell under another condition. ``summaries.py`` builds those vectors by
globbing two directories and subtracting elementwise, which is correct only
while the two globs enumerate the same (family, budget) sequence. Here the
pairing is by key, and any disagreement with positional pairing is reported.

The permutation experiment (``sdrefu_*``) has no generator in the package yet,
so its family-level table is computed here too, under the same rules:

    G(arm) = [recall(arm) - recall(stl)] measured
           - [recall(arm) - recall(stl)] null

for both nulls, as the quantity that says whether the uniform-history gain is
carried by the auxiliary activity values or by the act of selecting on them.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MACROS = ROOT / "report" / "generated" / "macros.tex"

# condition -> how to find that condition's artefact for a (family, budget).
SOURCES = {
    "withheld": "win32_{family}_b{budget}.json",
    "uniform": "winleak_{family}_b{budget}.json",
    "uniform_q0": "sdrefu_{family}_b{budget}_q0.json",
    "uniform_shuf": "sdrefu_{family}_b{budget}_shuf.json",
}


# ---------------------------------------------------------------------------
# artefacts

def cell_key(path: Path) -> tuple[str, int]:
    m = re.search(r"_(?P<family>[a-z0-9]+)_b(?P<budget>\d+)", path.name)
    return m["family"], int(m["budget"])


def load_condition(pattern: str) -> dict[tuple[str, int], dict]:
    out = {}
    for path in sorted(RESULTS.glob(pattern.format(family="*", budget="*"))):
        out[cell_key(path)] = json.loads(path.read_text(encoding="utf-8"))
    return out


def contrast(art: dict, arm: str, baseline: str = "stl") -> np.ndarray | None:
    """Per-campaign recall difference, in points. 128 values per cell."""
    arms = art.get("arms") or {}
    if arm not in arms or baseline not in arms:
        return None
    a, b = arms[arm].get("recall"), arms[baseline].get("recall")
    if a is None or b is None:
        return None
    return (np.asarray(a, float) - np.asarray(b, float)) * 100.0


def cell_means(cond: dict[tuple[str, int], dict], arm: str) -> dict:
    out = {}
    for key, art in cond.items():
        v = contrast(art, arm)
        if v is not None:
            out[key] = float(v.mean())
    return out


def to_families(cells: dict) -> dict[str, float]:
    """Budget cells -> one value per family. Two budgets on one family are not
    independent, so they are averaged rather than counted twice."""
    agg = defaultdict(list)
    for (family, _budget), v in cells.items():
        agg[family].append(v)
    return {f: float(np.mean(v)) for f, v in sorted(agg.items())}


# ---------------------------------------------------------------------------
# statistics

def summarize(values, *, unit: str, name: str, source: str, note: str = "") -> dict:
    x = np.asarray(list(values), dtype=float)
    n = len(x)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    zeros = int((x == 0).sum())
    nz = np.abs(x[x != 0])
    ties = int(len(nz) - len(np.unique(nz)))
    # With a tie or a zero the exact distribution no longer applies, and the
    # three conventions on offer disagree in the third decimal. Which one was
    # used stops being a detail the moment a p lands next to 0.05, so all of
    # them are recorded and the headline p names its rule.
    alternatives = {}
    if not np.any(x != 0):
        p, method = 1.0, "degenerate (all differences zero)"
    else:
        for label, kw in (("exact", dict(method="exact")),
                          ("approx", dict(method="approx", correction=False)),
                          ("approx_cc", dict(method="approx", correction=True)),
                          ("scipy_auto", dict())):
            try:
                alternatives[label] = float(wilcoxon(x, **kw).pvalue)
            except Exception as exc:  # scipy refuses exact for some inputs
                alternatives[label] = None
                alternatives[f"{label}_error"] = str(exc)
        if zeros == 0 and ties == 0 and n <= 50:
            p, method = alternatives["exact"], "exact"
        else:
            p = alternatives["approx_cc"]
            method = (f"normal approximation with continuity correction "
                      f"({zeros} zeros, {ties} ties make the exact "
                      "distribution inapplicable)")
    return {
        "name": name, "unit": unit, "n": n, "estimate": mean, "sem": sem,
        "ci95": [mean - half, mean + half],
        "positive": int((x > 0).sum()),
        # The rank test is the headline because the family-level vectors are
        # six points and nothing vouches for their normality. The paired t is
        # reported beside it because the interval above *is* the t interval,
        # and because a rank test on six points cannot go below 0.031 however
        # large the effect.
        "p": p, "p_method": method, "p_alternatives": alternatives,
        "p_ttest": (float(ttest_1samp(x, 0.0).pvalue) if n > 1 else None),
        "zeros": zeros, "ties": ties,
        "p_floor": min(1.0, 2.0 ** (1 - n)),
        "values": [float(v) for v in x],
        "source": source, "note": note,
    }


# ---------------------------------------------------------------------------
# rendering the permutation result the manuscript had no generator for

FAMILY_LABEL = {"cyp": "CYP", "dpp": "DPP", "hdac": "HDAC",
                "jak": "JAK", "mmp": "MMP", "pi3k": "PI3K"}


def emit_design_family(records: list[dict], conds: dict, tex_dir: Path) -> list[Path]:
    """The C4 history-design comparison at the family level, for the main text.

    The manuscript's own inference unit is the family: the two budget cells of
    a family share one historical draw, which the run digests demonstrate. The
    main text therefore carries one row per family, and the twelve-cell grid
    moves to the supplement, where `design_table_check` still verifies every
    typed value against the artefacts.

    `Cells >0` is retained because it is the per-cell consistency count the
    text refers to, and it is computed here rather than typed.
    """
    by_name = {r["name"]: r for r in records}
    cols = ["MtWithheld", "MtUniform", "PooledWithheld", "PooledUniform"]
    if any(c not in by_name for c in cols):
        return []
    fams = sorted(FAMILY_LABEL)

    # Coverage is a family property: the history is 50 per auxiliary target on
    # three auxiliaries, against the family's pool, and both budgets share it.
    coverage = {}
    for (fam, _budget), art in conds["uniform"].items():
        coverage[fam] = 3 * art["history"] / art["n_pool"]

    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
            " & & \\multicolumn{2}{c}{\\mt{} $-$ \\stl{}}"
            " & \\multicolumn{2}{c}{\\pooled{} $-$ \\stl{}} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Family & Coverage & withheld & uniform & withheld & uniform \\\\",
            "\\midrule"]
    for i, fam in enumerate(fams):
        cells = " & ".join(f"${by_name[c]['values'][i]:+.2f}$" for c in cols)
        rows.append(f"{FAMILY_LABEL[fam]} & ${coverage[fam] * 100:.0f}\\%$ "
                    f"& {cells} \\\\")
    rows.append("\\midrule")
    rows.append("Mean over families & & "
                + " & ".join(f"${by_name[c]['estimate']:+.2f}$" for c in cols)
                + " \\\\")
    rows.append("Families $>0$ & & "
                + " & ".join(f"{by_name[c]['positive']}/{by_name[c]['n']}"
                             for c in cols) + " \\\\")
    # The per-cell counts, from the cells themselves rather than from a tally
    # typed beside a table that no longer lists them.
    per_cell = []
    for arm, cond in (("mt", "withheld"), ("mt", "uniform"),
                      ("pooled", "withheld"), ("pooled", "uniform")):
        vals = cell_means(conds[cond], arm)
        per_cell.append(f"{sum(1 for v in vals.values() if v > 0)}/{len(vals)}")
    rows.append("Cells $>0$ & & " + " & ".join(per_cell) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    path = tex_dir / "table_design_family.tex"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [path]


def emit_permutation(records: list[dict], tex_dir: Path) -> list[Path]:
    """The six-family null comparison, as a supplement table and macros.

    This result was computed in a session note and typed into the manuscript by
    hand, which is the one place in the evidence package where a number had no
    generator behind it. Every value below comes from the same `summarize`
    record that the audit checks, including the per-family column, which is the
    `values` array those records already carry.

    Both nulls are rendered. The permutation null preserves each auxiliary
    target's marginal distribution and destroys only the activity--compound
    correspondence; the Gaussian null destroys the marginal too. Reporting the
    difference between them is what distinguishes "the correspondence mattered"
    from "the shape of the auxiliary column mattered".
    """
    by_name = {r["name"]: r for r in records}
    needed = ["MtUniform", "MtUnderShuf", "MtGvalueShuf",
              "PooledUniform", "PooledUnderShuf", "PooledGvalueShuf"]
    if any(n not in by_name for n in needed):
        return []
    fams = sorted(FAMILY_LABEL)

    rows = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
            " & \\multicolumn{3}{c}{\\mt{} $-$ \\stl{}}"
            " & \\multicolumn{3}{c}{\\pooled{} $-$ \\stl{}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            "Family & measured & permuted & difference"
            " & measured & permuted & difference \\\\", "\\midrule"]
    for i, fam in enumerate(fams):
        cells = [f"${by_name[n]['values'][i]:+.2f}$" for n in needed]
        rows.append(f"{FAMILY_LABEL[fam]} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    mean = [f"${by_name[n]['estimate']:+.2f}$" for n in needed]
    rows.append("Mean & " + " & ".join(mean) + " \\\\")
    sign = [f"{by_name[n]['positive']}/{by_name[n]['n']}" for n in needed]
    rows.append("Families $>0$ & " + " & ".join(sign) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    table = tex_dir / "table_permutation.tex"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")

    lines = ["% Generated by scripts/family_inference_audit.py -- do not edit.",
             "% Six-family null comparison under uniformly sampled histories.",
             ""]
    for name in ("MtUniform", "MtUnderShuf", "MtGvalueShuf", "MtUnderQ0",
                 "MtGvalueQ0", "MtNullDifference", "PooledUniform",
                 "PooledUnderShuf", "PooledGvalueShuf", "PooledUnderQ0",
                 "PooledGvalueQ0", "PooledNullDifference"):
        r = by_name.get(name)
        if r is None:
            continue
        lo, hi = r["ci95"]
        # A LaTeX control sequence is letters only, so the `Q0` in the
        # Gaussian-null record names has to be spelled out. Without this the
        # file loads and the document then fails at \begin{document} with an
        # error that points at the macro, not at the digit.
        name = name.replace("0", "Zero")
        for pre, val in (("perm", f"${r['estimate']:+.2f}$"),
                         ("permSem", f"${r['sem']:.2f}$"),
                         ("permCi", f"$[{lo:+.2f},{hi:+.2f}]$"),
                         ("permPos", f"{r['positive']}/{r['n']}"),
                         ("permP", f"${r['p']:.3f}$"),
                         ("permPt", f"${r['p_ttest']:.3f}$")):
            lines.append(f"\\newcommand{{\\{pre}{name}}}{{{val}}}")
        lines.append("")
    # The floor is a property of n, not of any one record, and it is the
    # sentence a reader needs next to every 0.031 in the table.
    floor = min((r["p_floor"] for r in records if r["n"] == 6), default=None)
    if floor is not None:
        lines.append(f"\\newcommand{{\\permPFloor}}{{${floor:.3f}$}}")
    macros = tex_dir / "macros_permutation.tex"
    macros.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [table, macros]


# ---------------------------------------------------------------------------
# the manuscript's rendered macros

def read_macros(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", text,
                           re.MULTILINE))


def as_number(raw: str) -> float | None:
    m = re.search(r"[-+]?\d+\.?\d*", raw.replace("$", "").replace("\\", ""))
    return float(m.group()) if m else None


def as_interval(raw: str) -> list[float] | None:
    nums = re.findall(r"[-+]?\d+\.?\d*", raw.replace("$", ""))
    return [float(nums[0]), float(nums[1])] if len(nums) >= 2 else None


def check_macro(rec: dict, base: str, macros: dict, problems: list) -> dict:
    """Compare one summary against the four macros the manuscript renders."""
    checks = {}
    want = {
        f"val{base}": (rec["estimate"], as_number),
        f"sem{base}": (rec["sem"], as_number),
        f"n{base}": (rec["n"], as_number),
        f"p{base}": (rec["p"], as_number),
    }
    for macro, (value, parse) in want.items():
        if macro not in macros:
            continue
        got = parse(macros[macro])
        places = 0 if macro.startswith("n") else (3 if macro.startswith("p") else 2)
        ok = got is not None and abs(got - round(value, places)) < 10 ** -places / 2
        checks[macro] = {"rendered": macros[macro], "recomputed": round(value, places),
                         "match": bool(ok)}
        if not ok:
            why = ""
            if macro.startswith("p") and (rec["ties"] or rec["zeros"]):
                spread = [v for k, v in rec["p_alternatives"].items()
                          if isinstance(v, float)]
                why = (f" -- {rec['ties']} ties and {rec['zeros']} zeros make "
                       "the exact test inapplicable, and the conventions span "
                       f"{min(spread):.3f}-{max(spread):.3f}; the rendered "
                       "value is scipy's default, this audit reports the "
                       "continuity-corrected one")
            problems.append(f"{macro}: manuscript says {macros[macro]}, "
                            f"artefacts give {value:.{places}f}{why}")
    if f"ci{base}" in macros:
        got = as_interval(macros[f"ci{base}"])
        want_ci = [round(v, 2) for v in rec["ci95"]]
        ok = got is not None and all(abs(a - b) < 0.005 for a, b in zip(got, want_ci))
        checks[f"ci{base}"] = {"rendered": macros[f"ci{base}"],
                               "recomputed": want_ci, "match": bool(ok)}
        if not ok:
            problems.append(f"ci{base}: manuscript says {macros[f'ci{base}']}, "
                            f"artefacts give [{want_ci[0]:+.2f},{want_ci[1]:+.2f}]")
    if f"pos{base}" in macros:
        rendered = macros[f"pos{base}"]
        want_pos = f"{rec['positive']}/{rec['n']}"
        ok = rendered.strip() == want_pos
        checks[f"pos{base}"] = {"rendered": rendered, "recomputed": want_pos,
                                "match": bool(ok)}
        if not ok:
            problems.append(f"pos{base}: manuscript says {rendered}, "
                            f"artefacts give {want_pos}")
    return checks


# ---------------------------------------------------------------------------
# the audit

def pairing_report(conds: dict) -> dict:
    """Do all four conditions cover the same cells, with the same geometry?"""
    keys = {name: set(c) for name, c in conds.items()}
    common = set.intersection(*keys.values())
    rec = {"cells": sorted(f"{f}_b{b}" for f, b in common), "problems": []}
    for name, k in keys.items():
        missing = common ^ k
        if missing:
            rec["problems"].append(
                f"{name} covers {sorted(f'{f}_b{b}' for f in [m[0] for m in missing] for b in [m[1] for m in missing])[:4]} "
                "differently from the others")
    # Same geometry per cell: a paired difference between cells of different
    # size or different repeat counts is not a paired difference.
    for key in sorted(common):
        shapes = {}
        for name, cond in conds.items():
            art = cond[key]
            n_campaigns = len(art["arms"]["stl"]["recall"])
            shapes[name] = (art["budget"], art["k"], art["history"],
                            art["repeats"], art["n_pool"], n_campaigns)
        if len({v for v in shapes.values()}) != 1:
            rec["problems"].append(f"{key[0]}_b{key[1]}: geometry differs "
                                   f"across conditions: {shapes}")
    # The uniform/withheld pair must differ in the history design and in
    # nothing else; the two nulls must share the uniform design.
    for key in sorted(common):
        if conds["withheld"][key]["history_excludes_top"] is not True:
            rec["problems"].append(f"{key[0]}_b{key[1]}: withheld artefact does "
                                   "not carry history_excludes_top")
        for name in ("uniform", "uniform_q0", "uniform_shuf"):
            if conds[name][key]["history_excludes_top"] is not False:
                rec["problems"].append(f"{key[0]}_b{key[1]}: {name} artefact "
                                       "carries history_excludes_top")
    rec["clean"] = not rec["problems"]
    return rec


def design_table_check(tex: Path, conds: dict, problems: list) -> dict:
    """The 48 hand-typed cells of the C4 table, against the artefacts.

    That table carries a ``% summaries-exempt`` marker: its per-cell grid is
    typed into the manuscript rather than generated, because generating it
    would duplicate the null-control table on two shared rows. The exemption
    is from *generation*, not from being right, and nothing checked it. The
    family means underneath it are the paper's cross-family evidence, so an
    error in a row would be an error in the evidence.
    """
    rec = {"rows": 0, "cells": 0, "problems": [], "found_in": None}
    marker = "Cell & Coverage & withheld & uniform"
    # The family-level summary is in the main text and the twelve-cell grid in
    # the supplement, so both are searched. A marker found in neither is a
    # problem, not a silent skip: the previous version returned quietly, which
    # would have retired this check the moment the table moved.
    paths = [tex] if isinstance(tex, Path) else list(tex)
    hit = next((q for q in paths if q.exists()
                and marker in q.read_text(encoding="utf-8")), None)
    if hit is None:
        problems.append("the C4 per-cell grid was not found in "
                        + ", ".join(q.name for q in paths)
                        + "; its 48 typed values are no longer checked")
        return rec
    rec["found_in"] = hit.name
    text = hit.read_text(encoding="utf-8")
    body = text.split(marker)[1].split("\\midrule")[1]
    order = [("mt", "withheld"), ("mt", "uniform"),
             ("pooled", "withheld"), ("pooled", "uniform")]
    for line in body.splitlines():
        cols = [c.strip() for c in line.split("&")]
        if len(cols) != 6 or not re.match(r"^[A-Z0-9]+ \$b\d+\$$", cols[0]):
            continue
        rec["rows"] += 1
        family = cols[0].split()[0].lower()
        budget = int(re.search(r"\d+", cols[0].split()[1]).group())
        coverage = int(re.search(r"\d+", cols[1]).group())
        typed = [float(re.search(r"[-+][\d.]+", c).group()) for c in cols[2:]]
        for (arm, cond), value in zip(order, typed):
            art = conds[cond].get((family, budget))
            if art is None:
                rec["problems"].append(f"{family} b{budget}: no {cond} artefact")
                continue
            got = float(contrast(art, arm).mean())
            rec["cells"] += 1
            if round(got, 2) != round(value, 2):
                rec["problems"].append(
                    f"{family} b{budget} {arm}/{cond}: table {value:+.2f}, "
                    f"artefacts {got:+.2f}")
        art = conds["uniform"].get((family, budget))
        if art is not None:
            n_t = len(art.get("per_target")
                      or (art.get("provenance") or {}).get("targets") or [])
            got = art["history"] * (n_t - 1) / art["n_pool"]
            if abs(round(got * 100) - coverage) > 1:
                rec["problems"].append(
                    f"{family} b{budget}: coverage {coverage}% typed, "
                    f"{got:.1%} in the artefact")
    problems += rec["problems"]
    return rec


def positional_pairing_check(a: dict, b: dict) -> dict:
    """Does subtracting two sorted globs elementwise agree with pairing by key?

    `summaries.py` builds its cell-level vectors that way. It is correct here,
    and silently wrong the moment one condition gains or loses a cell.
    """
    ka, kb = sorted(a), sorted(b)
    by_key = np.array([a[k] - b[k] for k in ka if k in b])
    positional = np.array([a[x] - b[y] for x, y in zip(ka, kb)])
    agree = (len(ka) == len(kb) and ka == kb
             and np.allclose(by_key, positional))
    return {"aligned": bool(agree), "n_by_key": int(len(by_key)),
            "n_positional": int(len(positional))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=RESULTS / "family_inference_audit.json")
    p.add_argument("--macros", type=Path, default=MACROS)
    p.add_argument("--tex-dir", type=Path, default=None,
                   help="also render the six-family permutation result as a "
                        "supplement table and macros (report/generated)")
    p.add_argument("--paper", type=Path, nargs="+",
                   default=[ROOT / "report" / "paper.tex",
                            ROOT / "report" / "supplement.tex"],
                   help="the manuscript files, for the hand-typed C4 grid")
    args = p.parse_args(argv)

    conds = {name: load_condition(pat) for name, pat in SOURCES.items()}
    for name, cond in conds.items():
        print(f"{name:<14s} {len(cond)} cells  ({SOURCES[name]})")
    macros = read_macros(args.macros)
    problems: list[str] = []

    pairing = pairing_report(conds)
    print(f"\npairing: {len(pairing['cells'])} cells common to all four "
          f"conditions, geometry {'consistent' if pairing['clean'] else 'INCONSISTENT'}")
    for prob in pairing["problems"]:
        print(f"    ! {prob}")
    problems += pairing["problems"]

    table = design_table_check(args.paper, conds, problems)
    if table["rows"]:
        print(f"C4 table ({table['found_in']}): {table['cells']} typed cells "
              f"in {table['rows']} rows"
              + (" all reproduce" if not table["problems"] else " checked"))
        for prob in table["problems"]:
            print(f"    ! {prob}")

    records: list[dict] = []
    campaigns = {len(art["arms"]["stl"]["recall"])
                 for cond in conds.values() for art in cond.values()}

    # --- the manuscript's family-level statistics --------------------------
    for arm in ("mt", "pooled"):
        cells = {name: cell_means(cond, arm) for name, cond in conds.items()}
        fams = {name: to_families(c) for name, c in cells.items()}
        label = arm.capitalize() if arm == "pooled" else "Mt"

        for cond_name, base in (("uniform", f"{label}Uniform"),
                                ("withheld", f"{label}Withheld")):
            rec = summarize(fams[cond_name].values(), unit="protein family",
                            name=base, source=SOURCES[cond_name],
                            note=f"{arm} - stl, {cond_name} history")
            rec["families"] = fams[cond_name]
            rec["macros"] = check_macro(rec, base, macros, problems)
            records.append(rec)

        keys = sorted(fams["uniform"])
        rec = summarize([fams["uniform"][k] - fams["withheld"][k] for k in keys],
                        unit="protein family", name=f"{label}DesignChange",
                        source="winleak_* vs win32_*",
                        note=f"{arm} - stl, uniform minus withheld")
        rec["families"] = {k: fams["uniform"][k] - fams["withheld"][k] for k in keys}
        rec["macros"] = check_macro(rec, f"{label}DesignChange", macros, problems)
        records.append(rec)

        ckeys = sorted(set(cells["uniform"]) & set(cells["withheld"]))
        rec = summarize([cells["uniform"][k] - cells["withheld"][k] for k in ckeys],
                        unit="configuration", name=f"{label}DesignChangeCell",
                        source="winleak_* vs win32_*",
                        note="secondary figure: two budgets on one family are "
                             "not independent")
        rec["pairing"] = positional_pairing_check(cells["uniform"],
                                                  cells["withheld"])
        if not rec["pairing"]["aligned"]:
            problems.append(f"{label}DesignChangeCell: positional pairing does "
                            "not agree with pairing by (family, budget)")
        rec["macros"] = check_macro(rec, f"{label}DesignChangeCell", macros, problems)
        records.append(rec)

        # --- the permutation experiment, same rules ------------------------
        for null, null_name in (("uniform_q0", "Q0"), ("uniform_shuf", "Shuf")):
            if not fams[null]:
                continue
            keys = sorted(set(fams["uniform"]) & set(fams[null]))
            rec = summarize([fams["uniform"][k] - fams[null][k] for k in keys],
                            unit="protein family",
                            name=f"{label}Gvalue{null_name}",
                            source=f"winleak_* vs {SOURCES[null]}",
                            note="value of the auxiliary activities: measured "
                                 "minus " + ("Gaussian null" if null == "uniform_q0"
                                             else "permutation null"))
            rec["families"] = {k: fams["uniform"][k] - fams[null][k] for k in keys}
            records.append(rec)
            # The arm's own level under the null, which is what "pooled pays a
            # selection penalty even with no information" is read off.
            rec_null = summarize(fams[null].values(), unit="protein family",
                                 name=f"{label}Under{null_name}",
                                 source=SOURCES[null],
                                 note=f"{arm} - stl under the null alone")
            rec_null["families"] = fams[null]
            records.append(rec_null)
        if fams["uniform_shuf"] and fams["uniform_q0"]:
            keys = sorted(fams["uniform_shuf"])
            # G under the permutation minus G under the Gaussian null, which
            # is the same number as (Gaussian - permutation) at arm level with
            # the sign the quantity is quoted in: how much of the measured
            # value survives destroying the column's *shape* as well as its
            # information. Near zero means the nulls agree, so the value is
            # information and not a distributional artefact.
            rec = summarize([fams["uniform_q0"][k] - fams["uniform_shuf"][k]
                             for k in keys], unit="protein family",
                            name=f"{label}NullDifference",
                            source="sdrefu_*_shuf vs sdrefu_*_q0",
                            note="G(permutation) - G(Gaussian): is the null a "
                                 "distributional artefact?")
            records.append(rec)

    # --- report -------------------------------------------------------------
    print(f"\neach budget cell averages {sorted(campaigns)} campaigns; "
          "no cross-family statistic below uses that as its n")
    print(f"\n{'quantity':<24}{'unit':<18}{'n':>3}{'estimate':>10}{'sem':>8}"
          f"{'95% CI':>20}{'sign':>7}{'p':>8}{'floor':>7}{'p(t)':>8}")
    for rec in records:
        lo, hi = rec["ci95"]
        print(f"{rec['name']:<24}{rec['unit']:<18}{rec['n']:>3}"
              f"{rec['estimate']:>+10.2f}{rec['sem']:>8.2f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}"
              f"{f'{rec['positive']}/{rec['n']}':>7}"
              f"{rec['p']:>8.3f}{rec['p_floor']:>7.3f}"
              f"{rec['p_ttest']:>8.3f}")

    bad_unit = [r for r in records
                if r["unit"] == "protein family" and r["n"] != 6]
    for rec in bad_unit:
        problems.append(f"{rec['name']}: claims the family unit but n={rec['n']}")

    checked = sum(len(r.get("macros", {})) for r in records)
    print(f"\n{checked} rendered macros checked against the artefacts")
    if problems:
        print(f"{len(problems)} problems:")
        for prob in problems:
            print(f"    ! {prob}")
    else:
        print("no disagreement between the manuscript and the artefacts")

    args.out.write_text(json.dumps(
        {"campaigns_per_cell": sorted(campaigns), "pairing": pairing,
         "design_table": table,
         "macros_checked": checked, "problems": problems,
         "clean": not problems, "records": records}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.tex_dir:
        for path in (emit_permutation(records, args.tex_dir)
                     + emit_design_family(records, conds, args.tex_dir)):
            print(f"wrote {path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

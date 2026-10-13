"""One inventory of every campaign experiment, built from the run records.

    python scripts/experiment_inventory.py
    python scripts/experiment_inventory.py --json results/experiment_inventory.json
    python scripts/experiment_inventory.py --tex report/inventory.tex

The manuscript twice stated campaign counts that disagreed with each other --
"eight seeds per target" in the Methods against "32 repeats per target" in a
caption -- because the counts were written from the launch scripts and from
memory instead of from the artefacts. Both numbers were real; they belonged to
different experiments, and one table mixed two of them in adjacent rows.

This script is the single source of truth. It reads only stored artefacts,
never re-runs anything, and reports for each experiment group:

* how many seeds per held-out target, how many targets, and the product --
  the number of campaigns behind each cell;
* the history design actually used (``history_excludes_top``), which is a
  protocol setting the paper argues about and must therefore not be guessed;
* history size, budget range, acquisition, arms, and the block count;
* whether every artefact in the group carries a clean runtime audit.

A group whose artefacts disagree on any of these is reported as a conflict
rather than averaged away, because that is exactly the failure this exists to
catch.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# What each discovered prefix feeds in the paper. A prefix that is not named
# here is still inventoried, as "(not cited)" -- the inventory is built by
# *scanning* `results/`, never from a list of groups written by hand.
#
# The first version of this script did use a hand-written list, and it omitted
# `sdref_*`, which is the group the null-control table is actually built from.
# The inventory then "confirmed" a campaign count for that table which came
# from a different experiment. An inventory that can miss an experiment is not
# an inventory, so discovery is now exhaustive by construction.
FEEDS = {
    "sdref": "null-control table (all rows)",
    "sdscale": "matched-uncertainty baseline and the scale sweep",
    "auxq": "auxiliary-quality ladder (appendix)",
    "shuf": "permutation null (CYP, both designs)",
    "win32": "history-design table, withheld columns",
    "winleak": "history-design table, uniform columns",
    "sdrefu": "uniform-history null controls, six families (both nulls)",
    "avoid": "history-avoidance arm against the pooled model",
    "avoidw": "avoidance-weight sweep, the grid the calibration reads",
    "avoidcal": "calibrated avoidance, run directly at the matched ratio",
    "xfam_transfer": "decision-rule table, transfer option",
    "xfam_seed": "decision-rule table, seeding option",
    "xfam_transfer_excl": "decision-rule table, transfer, withheld",
    "xfam_seed_excl": "decision-rule table, seeding, withheld",
}

# Prefixes whose artefacts are superseded or exploratory and are not inventoried
# as experiments. Each is named, so that excluding it is a decision on the
# record rather than an accident of globbing.
SKIP = ("_", "acq_", "reach_", "pf_", "win_", "ladder", "prior_", "sim_")

FIELDS = ("repeats", "history", "history_excludes_top", "acquisition",
          "batch", "k", "pile_frac")


def one(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    # A few things in `results/` are JSON lists rather than campaign artefacts
    # -- `summaries.json`, `experiment_inventory.json`, the LLM call logs.
    # Discovery is exhaustive on purpose, so the inventory reads them too; it
    # has to answer "not a campaign grid" rather than raise.
    if not isinstance(d, dict):
        d = {}
    arms = d.get("arms") or {}
    # Sweep artefacts (`cross_family_scan`, `seed_rho_curve`, `bad_start_gate`)
    # write `arms` as the *list of names* they ran, with the per-campaign
    # vectors kept per sweep point rather than per arm. Discovery is exhaustive
    # by design, so the inventory meets them, and a campaign-level scan of a
    # tree containing one used to end in an AttributeError rather than a
    # report. They carry no campaign vector at this level, which the row below
    # records as `campaigns: None` -- the same answer as for any other artefact
    # that is not a campaign grid.
    by_arm = arms if isinstance(arms, dict) else {}
    a = next((v for v in by_arm.values()
              if isinstance(v, dict) and "recall" in v), None)
    prov = d.get("provenance") or {}
    audit = d.get("audit") or {}
    # Some groups were originally run without the audit flag. `rerun_audited.py`
    # re-runs those from their own recorded parameters into `results/audited/`
    # and checks that the numbers are unchanged; the originals are left alone.
    # A group is only fully accounted for if every artefact is audited either
    # in place or by such a re-run, so both are counted here.
    twin = RESULTS / "audited" / path.name
    twin_audit = {}
    if twin.exists() and twin != path:
        try:
            twin_audit = json.loads(twin.read_text(encoding="utf-8")).get(
                "audit") or {}
        except (OSError, ValueError):
            twin_audit = {}
    return {
        "file": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "block": d.get("family"),
        "budget": d.get("budget"),
        "campaigns": len(a["recall"]) if a else None,
        "targets": len(d.get("per_target") or prov.get("targets") or []),
        "arms": tuple(sorted(arms)),
        "source": prov.get("source"),
        "audit_enabled": bool(audit.get("enabled")),
        "audit_clean": bool(audit.get("clean")),
        "audit_clean_on_rerun": bool(twin_audit.get("clean")),
        **{f: d.get(f) for f in FIELDS},
    }


def summarise(name: str, pattern: str, feeds: str,
              only_design: bool | None = None) -> dict | None:
    # Patterns are relative to RESULTS, never to the working directory, so
    # that the inventory of a results tree is a pure function of that tree.
    files = sorted(Path(p) for p in globmod.glob(str(RESULTS / pattern)))
    rows = [one(f) for f in files]
    rows = [r for r in rows if r["campaigns"]]
    if only_design is not None:
        # The history design is the protocol axis the paper argues about, so a
        # group holding both is reported as two rows rather than as one row
        # with a conflict. Every other setting must still agree within a row.
        rows = [r for r in rows if r["history_excludes_top"] is only_design]
        name = f"{name} [{'withheld' if only_design else 'uniform'}]"
    if not rows:
        return None

    def uniq(key):
        return sorted({r[key] for r in rows}, key=str)

    # Protocol settings only. Which arms a cell ran is not a conflict: a sweep
    # legitimately contains a single-arm rung and a multi-arm reference run.
    # What must not vary inside a group is how the campaign was set up.
    conflicts = [k for k in FIELDS + ("targets", "campaigns")
                 if len(uniq(k)) > 1 and k != "budget"]
    seeds = uniq("repeats")
    targets = uniq("targets")
    camps = uniq("campaigns")
    budgets = uniq("budget")
    return {
        "group": name, "feeds": feeds, "pattern": f"results/{pattern}",
        "n_artefacts": len(rows),
        "blocks": len({r["block"] for r in rows}),
        "seeds_per_target": seeds, "targets": targets,
        "campaigns_per_configuration": camps,
        "configurations": len(rows),
        "distinct_budgets": len(budgets),
        "total_campaigns": int(sum(r["campaigns"] for r in rows)),
        "seeds_times_targets_consistent": all(
            r["repeats"] * r["targets"] == r["campaigns"] for r in rows),
        "history": uniq("history"),
        "history_design": ["withheld" if v else "uniform"
                           for v in uniq("history_excludes_top")],
        "budgets": budgets,
        "acquisition": uniq("acquisition"),
        "arms": uniq("arms"),
        "source": uniq("source"),
        "audited": sum(r["audit_enabled"] for r in rows),
        "audit_clean": sum(r["audit_clean"] for r in rows),
        "audit_clean_on_rerun": sum(r["audit_clean_on_rerun"] for r in rows),
        "audit_accounted": sum(r["audit_clean"] or r["audit_clean_on_rerun"]
                               for r in rows),
        "conflicts": conflicts,
    }


def fmt(v):
    if isinstance(v, list):
        if len(v) == 1:
            return fmt(v[0])
        if all(isinstance(x, (int, float)) and x is not None for x in v):
            return f"{min(v)}-{max(v)}"
        return ", ".join(str(x) for x in v)
    return "--" if v is None else str(v)


def discover() -> list[tuple[str, str, str]]:
    """Every campaign-artefact group under results/, found by scanning.

    A group is a directory of artefacts (``results/xfam_seed/*.json``) or a
    filename prefix up to the first block name (``results/sdref_jak_0.0.json``
    -> ``sdref``). Anything carrying per-campaign recall counts as a campaign
    artefact; anything else is ignored.
    """
    groups: dict[str, str] = {}
    for f in sorted((RESULTS).rglob("*.json")):
        rel = f.relative_to(RESULTS)
        if any(rel.parts[0].startswith(s) for s in SKIP):
            continue
        if len(rel.parts) > 1:
            name = rel.parts[0]
            pattern = f"{name}/*.json"
        else:
            stem = rel.stem
            if "_" not in stem:
                continue
            name = stem.split("_")[0]
            pattern = f"{name}_*.json"
        groups.setdefault(name, pattern)

    out = []
    for name, pattern in sorted(groups.items()):
        out.append((name, pattern, FEEDS.get(name, "(not cited)")))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--tex", type=Path, default=None)
    p.add_argument("--cited-only", action="store_true",
                   help="only the groups the paper draws numbers from")
    args = p.parse_args(argv)

    specs = discover()
    if args.cited_only:
        specs = [s for s in specs if s[0] in FEEDS]
    inv = []
    for spec in specs:
        g = summarise(*spec)
        if g is None:
            continue
        if "history_excludes_top" in g["conflicts"]:
            inv += [x for x in (summarise(*spec, only_design=d)
                                for d in (False, True)) if x]
        else:
            inv.append(g)
    if not inv:
        raise SystemExit("no campaign artefacts found; run the experiments "
                         "before building an inventory of them.")

    print("Experiment inventory, read from the stored run records\n")
    print("  A *configuration* is one stored run: one block, one budget, and "
          "for the\n  quality ladder one requested q. Campaigns per "
          "configuration is seeds per target\n  x held-out targets, and "
          "total = configurations x campaigns per configuration.\n")
    print(f"  {'group':<30}{'blocks':>7}{'configs':>8}{'seeds/tgt':>10}"
          f"{'targets':>8}{'camp/config':>12}{'total':>7}{'design':>10}"
          f"{'audit':>8}")
    print("  " + "-" * 100)
    for g in inv:
        print(f"  {g['group']:<30}{g['blocks']:>7}{g['configurations']:>8}"
              f"{fmt(g['seeds_per_target']):>10}{fmt(g['targets']):>8}"
              f"{fmt(g['campaigns_per_configuration']):>12}"
              f"{g['total_campaigns']:>7}"
              f"{fmt(g['history_design']):>10}"
              f"{str(g['audit_accounted']) + '/' + str(g['n_artefacts']):>8}"
              + ("  (re-run)" if g['audit_clean_on_rerun'] else ""))

    print(f"\n  {'':<30}{sum(g['total_campaigns'] for g in inv):>45} "
          f"campaigns in total")

    print("\n  budgets, history size and arms:")
    for g in inv:
        print(f"    {g['group']:<30}budget {fmt(g['budgets']):<14}"
              f"H {fmt(g['history']):<5} arms {fmt(g['arms'])}")

    bad = [g for g in inv if g["conflicts"]]
    inconsistent = [g for g in inv if not g["seeds_times_targets_consistent"]]
    unaudited = [g for g in inv if g["audit_accounted"] != g["n_artefacts"]]

    print()
    if bad:
        for g in bad:
            print(f"  CONFLICT in {g['group']}: artefacts disagree on "
                  f"{', '.join(g['conflicts'])}")
    if inconsistent:
        for g in inconsistent:
            print(f"  CONFLICT in {g['group']}: seeds x targets does not equal "
                  f"the stored campaign count")
    if unaudited:
        for g in unaudited:
            print(f"  NOTE: {g['group']} has "
                  f"{g['n_artefacts'] - g['audit_accounted']} artefact(s) "
                  f"with no clean runtime audit, in place or on re-run")
    if not (bad or inconsistent):
        print("  Every group is internally consistent, and seeds x targets "
              "equals the stored\n  campaign count in every artefact. These "
              "are the counts the manuscript must use.")

    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        lines = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
                 r"Experiment & Blocks & Configs & Seeds/target & Targets & "
                 r"\shortstack{Campaigns per\\configuration} & Total & "
                 r"History \\", r"\midrule"]
        for g in inv:
            lines.append(
                f"\\texttt{{{g['group'].replace('_', chr(92) + '_')}}} & "
                f"{g['blocks']} & {g['configurations']} & "
                f"{fmt(g['seeds_per_target'])} & {fmt(g['targets'])} & "
                f"{fmt(g['campaigns_per_configuration'])} & "
                f"{g['total_campaigns']} & "
                f"{fmt(g['history_design'])} \\\\")
        total = sum(g["total_campaigns"] for g in inv)
        lines += [r"\midrule",
                  f"Total & & & & & & {total} & \\\\",
                  r"\bottomrule", r"\end{tabular}"]
        args.tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.tex}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inv, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if (bad or inconsistent) else 0


if __name__ == "__main__":
    raise SystemExit(main())

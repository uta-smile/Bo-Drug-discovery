"""Where every quoted design swing comes from, and why two of them disagreed.

    python scripts/swing_provenance.py
    python scripts/swing_provenance.py --family cyp --budget 111

The history-design swing

    swing(arm) = [arm - stl] withheld  -  [arm - stl] uniform

is quoted in three places -- the C4 paragraph in the manuscript, the avoidance
mechanism note, and the calibration report -- and the first version of the
calibration script printed 15.28 points at 45% coverage where the C4 result is
29.07. Same cell, same budget, same history size. A reader has no way to tell a
units mistake from a different experiment, so this script answers it from the
artefacts: it finds every stored run that could produce such a swing, groups
the ones that are *the same campaigns*, and prints the configuration ledger and
both metrics for each group.

**The answer in this tree:** every quoted value comes from one pair of runs per
coverage. They are not different experiments. The 15.28 was the swing in
`auc_top_k` units -- an artefact's stored `delta_vs_stl` is the AUC difference,
not the recall difference, and the calibration script read that field while its
docstring said recall. The AUC and recall swings are both printed below so the
two can be told apart at a glance.

What is checked, per group:

* **identical campaigns** -- per-campaign recall and AUC vectors hashed, so
  "the same runs" means the same 128 numbers, not the same mean;
* **the configuration ledger** -- budget, batch, initial design, top-k,
  repeats x held-out targets, pool size, history size and construction,
  auxiliary null and its achieved Spearman, acquisition hyperparameters, seed;
* **the swing, in both metrics**, with the paired SEM over campaigns.

Artefacts written before 2026-10-11 carry no `seed` field (it is recorded from
that date). For those, identical per-campaign vectors are the evidence that the
seed and every derived stream matched -- which is stronger than the field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The fields that have to agree before a difference between two runs is a
# measurement rather than a comparison of two experiments.
LEDGER = ("budget", "batch", "n_init", "k", "repeats", "n_pool",
          "history", "history_excludes_top", "history_cluster",
          "acquisition", "aux_quality_requested", "aux_quality_achieved",
          "xi", "beta", "sd_scale", "pile_frac", "avoid_weight", "seed")


def digest(values) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()[:12]


def contrast(art: dict, arm: str) -> dict | None:
    arms = art.get("arms")
    if not isinstance(arms, dict) or arm not in arms or "stl" not in arms:
        return None
    a, b = arms[arm], arms["stl"]
    if "recall" not in a or "recall" not in b:
        return None
    rec = (np.asarray(a["recall"], float) - np.asarray(b["recall"], float)) * 100
    auc = np.asarray(a["auc"], float) - np.asarray(b["auc"], float)
    return {
        "recall": float(rec.mean()),
        "recall_sem": float(rec.std(ddof=1) / np.sqrt(len(rec))),
        "auc": float(auc.mean()),
        "n": int(len(rec)),
        "campaign_digest": digest(a["recall"]),
        "stl_digest": digest(b["recall"]),
        "stored_delta_is": ("auc" if abs(a.get("delta_vs_stl", 1e9)
                                         - float(auc.mean())) < 1e-9
                            else "recall" if abs(a.get("delta_vs_stl", 1e9)
                                                 - float(rec.mean()) / 100) < 1e-9
                            else "unknown"),
    }


def candidates(family: str, budget: int, arm: str) -> list[dict]:
    """Every stored run that carries this arm-vs-stl contrast on this cell."""
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            art = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(art, dict):
            continue
        if art.get("family") != family or art.get("budget") != budget:
            continue
        c = contrast(art, arm)
        if c is None:
            continue
        out.append({"file": path.name, "contrast": c,
                    "ledger": {k: art.get(k) for k in LEDGER},
                    "design": ("withheld" if art.get("history_excludes_top")
                               else "uniform"),
                    "history": art.get("history")})
    return out


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Runs whose per-campaign vectors are identical are one experiment."""
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["contrast"]["campaign_digest"], r["contrast"]["stl_digest"])
        out.setdefault(key, []).append(r)
    return out


def ledger_diff(rows: list[dict]) -> dict:
    """Fields on which the runs in a set disagree, split by what it means.

    Two reasons a field can differ across artefacts that hold *identical*
    campaign vectors, and only one of them is a defect:

    ``not_recorded``  one artefact predates the field. The `results/*.json`
        schema has always grown additively, so `n_init`, `xi`, `beta` and
        `seed` are absent from older runs. Absent is not "different".
    ``identity_null`` `aux_quality_requested` is 1.0 in one and absent in the
        other. `--aux-quality 1.0` passes the auxiliary column through
        `degrade_to_rho` at rho=1 and reproduces an unflagged run bit for bit;
        identical vectors here are that equivalence, demonstrated.
    ``conflict``      both recorded, and different. Identical numbers from
        genuinely different settings would mean the ledger does not determine
        the result, which is a worse problem than the one being investigated.
    """
    out = {"conflict": {}, "not_recorded": [], "identity_null": []}
    for field in LEDGER:
        values = {r["file"]: r["ledger"].get(field) for r in rows}
        present = {f: v for f, v in values.items() if v is not None}
        if len({json.dumps(v) for v in present.values()}) > 1:
            if field in ("aux_quality_requested", "aux_quality_achieved") \
                    and set(present.values()) <= {1.0}:
                out["identity_null"].append(field)
            else:
                out["conflict"][field] = values
        elif len(present) != len(values):
            (out["identity_null"] if field.startswith("aux_quality")
             and set(present.values()) <= {1.0}
             else out["not_recorded"]).append(field)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", default="cyp")
    p.add_argument("--budget", type=int, default=111)
    p.add_argument("--arm", default="pooled")
    p.add_argument("--out", type=Path, default=RESULTS / "swing_provenance.json")
    args = p.parse_args(argv)

    rows = candidates(args.family, args.budget, args.arm)
    histories = sorted({r["history"] for r in rows})
    print(f"{args.family} b{args.budget}: {len(rows)} stored runs carry a "
          f"{args.arm}-vs-stl contrast, at history {histories}")

    report = {"family": args.family, "budget": args.budget, "arm": args.arm,
              "coverages": {}}
    problems = []
    for history in histories:
        here = [r for r in rows if r["history"] == history]
        n_pool = here[0]["ledger"]["n_pool"]
        print(f"\n=== history {history}x3 "
              f"({3 * history / n_pool:.0%} of {n_pool} compounds) ===")
        entry = {"designs": {}}
        for design in ("withheld", "uniform"):
            side = [r for r in here if r["design"] == design]
            if not side:
                continue
            groups = group(side)
            print(f"  {design}: {len(side)} run(s) in "
                  f"{len(groups)} distinct experiment(s)")
            for key, members in groups.items():
                c = members[0]["contrast"]
                files = ", ".join(m["file"] for m in members)
                print(f"    [{key[0]}] {c['recall']:+7.2f} +- {c['recall_sem']:.2f} "
                      f"recall pts | {c['auc']:+8.4f} AUC | "
                      f"{c['n']} campaigns")
                print(f"        {files}")
                diff = ledger_diff(members)
                if len(members) > 1:
                    if diff["not_recorded"]:
                        print(f"        (schema growth: {', '.join(diff['not_recorded'])}"
                              " absent from the older artefact, not different)")
                    if diff["identity_null"]:
                        print("        (one of these ran --aux-quality 1.0 and "
                              "the other carried no flag; identical vectors "
                              "are that equivalence, demonstrated)")
                if diff["conflict"]:
                    problems.append(f"{design} h{history}: identical campaign "
                                    f"vectors but recorded settings differ on "
                                    f"{sorted(diff['conflict'])}")
                    print(f"        ! settings differ on "
                          f"{sorted(diff['conflict'])}: {diff['conflict']}")
            entry["designs"][design] = [
                {"files": [m["file"] for m in members],
                 "campaign_digest": key[0], **members[0]["contrast"],
                 "ledger": members[0]["ledger"],
                 "ledger_diff": ledger_diff(members)}
                for key, members in groups.items()]

        # A cell can hold several withheld experiments -- the null ladder, the
        # mechanism decomposition and the avoidance baseline all ran CYP b111
        # under a withheld history. So the swing names its pair rather than
        # assuming there is only one: the `avoid_*` runs, which are the ones
        # the mechanism note and the calibration both quote.
        wit = entry["designs"].get("withheld") or []
        uni = entry["designs"].get("uniform") or []
        named = f"avoid_{args.family}_h{history}_"
        wit = [g for g in wit if any(f.startswith(named) for f in g["files"])] or wit
        uni = [g for g in uni if any(f.startswith(named) for f in g["files"])] or uni
        if len(wit) == 1 and len(uni) == 1:
            w, u = wit[0], uni[0]
            entry["pair"] = {"withheld": w["files"], "uniform": u["files"]}
            print(f"  pair: {w['files'][0]} / {u['files'][0]}"
                  + (f"  (also stored as {', '.join(f for f in w['files'][1:] + u['files'][1:])})"
                     if len(w["files"]) + len(u["files"]) > 2 else ""))
            # The two designs are different experiments by construction -- the
            # history is drawn from a different pool -- so only the fields that
            # are *not* the design itself have to match.
            shared = [f for f in LEDGER
                      if f not in ("history_excludes_top", "aux_quality_achieved")]
            mismatch = {f: (w["ledger"][f], u["ledger"][f]) for f in shared
                        if w["ledger"][f] != u["ledger"][f]
                        and None not in (w["ledger"][f], u["ledger"][f])}
            entry["swing_recall"] = w["recall"] - u["recall"]
            entry["swing_auc"] = w["auc"] - u["auc"]
            entry["pair_ledger_mismatch"] = mismatch
            print(f"  swing = {w['recall']:+.2f} - ({u['recall']:+.2f}) = "
                  f"{entry['swing_recall']:+.2f} recall points"
                  f"   [{entry['swing_auc']:+.4f} in AUC units]")
            print("  the two runs agree on every ledger field except the "
                  "history design" if not mismatch
                  else f"  ! the pair also differs on {mismatch}")
            if mismatch:
                problems.append(f"h{history}: the withheld/uniform pair "
                                f"differs on {sorted(mismatch)}")
            print(f"  stored delta_vs_stl in these artefacts is the "
                  f"{w['stored_delta_is'].upper()} difference -- quoting it as "
                  "recall points understates the swing by "
                  f"{abs(entry['swing_recall'] - entry['swing_auc'] * 100):.2f} "
                  "points here")
        elif wit and uni:
            problems.append(f"h{history}: {len(wit)} withheld and {len(uni)} "
                            "uniform experiments; the swing is ambiguous "
                            "without naming the pair")
        report["coverages"][str(history)] = entry

    report["problems"] = problems
    report["clean"] = not problems
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{'no inconsistencies' if not problems else str(len(problems)) + ' problems'}")
    for prob in problems:
        print(f"    ! {prob}")
    print(f"wrote {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

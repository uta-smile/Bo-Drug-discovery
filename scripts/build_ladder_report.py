"""Build ``results/ladder_report.html`` from the ChEMBL and PMO ladder runs.

    python scripts/build_ladder_report.py

Reads whatever is present and says so when something is missing, because the
distinction this report exists to preserve is **"this arm lost" versus "this arm
never ran"**, and a gap in a table does not carry it.

Sources
-------
``results/chembl203_ladder_b50/``   summary.json + traces.jsonl (ChEMBL tier)
``results/pmo_ladder/``             pmo_results.json (23 PMO tasks)
``results/acquisition_headroom.json``  the gate that says whether the agentic
                                    action space can pay at all

Everything is inline: no CDN, no build step, no network at view time. The page
is a file in ``results/`` that survives being emailed.
"""

from __future__ import annotations

import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHEMBL_DIR = RESULTS / "chembl203_ladder_b50"
CHEMBL_AGENTIC_DIR = RESULTS / "chembl203_agentic_b50"
PMO_FILE = RESULTS / "pmo_ladder" / "pmo_results.json"
PMO_LLM_FILE = RESULTS / "pmo_ladder_llm" / "pmo_results.json"
HEADROOM = RESULTS / "acquisition_headroom.json"
OUT = RESULTS / "ladder_report.html"

# ---------------------------------------------------------------------------
# The ladder. Kept in sync with molbo.ladder.CATEGORY by a direct import, so
# the report cannot describe a set of arms the runners did not build.
# ---------------------------------------------------------------------------
from molbo.ladder import CATEGORY, LADDER_ARMS, NEEDS_MODEL  # noqa: E402

CATEGORY_ORDER = ["none", "bo", "llm", "bo+llm", "agentic", "ceiling"]

CATEGORY_META = {
    "none": (
        "No surrogate, no model",
        "Uniform draw from the untested pool. The floor every other arm has to "
        "clear before its machinery is worth discussing.",
    ),
    "bo": (
        "BO alone",
        "A GP over Tanimoto similarity plus a fixed acquisition. No model in "
        "the loop at any point.",
    ),
    "llm": (
        "LLM alone",
        "The model picks from a uniform random shortlist with no surrogate "
        "underneath it. Its control is <code>random</code> &mdash; the identical "
        "draw, picked from uniformly instead of by the model &mdash; so the gap "
        "between the two is the model's chemistry and nothing else.",
    ),
    "bo+llm": (
        "BO advised by the LLM",
        "The model narrows the library (<code>constraint</code>) or picks from "
        "BO's shortlist (<code>select</code>). <code>bo_random_select</code> is "
        "the control: the same shortlist, a uniform pick.",
    ),
    "agentic": (
        "BO steered by a controller",
        "The acquisition function and its hyperparameters are re-chosen every "
        "round. This is the SARA coupling. The scripted, random and round-robin "
        "policies are the controls the LLM policy has to beat.",
    ),
    "ceiling": (
        "Hindsight ceiling &mdash; not a method",
        "Reads the answer key and picks, each round, the acquisition whose top "
        "candidate has the best true value. A reference line saying how much "
        "room exists. It is greedy, so it is a loose bound and can be beaten by "
        "a policy with foresight.",
    ),
}

ACQ_ORDER = ["ei", "ucb", "pi", "greedy", "ts"]
ACQ_LABEL = {
    "ei": "expected improvement",
    "ucb": "upper confidence bound",
    "pi": "probability of improvement",
    "greedy": "posterior mean (exploit)",
    "ts": "Thompson sampling",
}

# Categorical slots 1-8 from the validated reference palette, light / dark.
SERIES = [
    ("#2a78d6", "#3987e5"),  # 1 blue
    ("#eb6834", "#d95926"),  # 2 orange
    ("#1baf7a", "#199e70"),  # 3 aqua
    ("#eda100", "#c98500"),  # 4 yellow
    ("#e87ba4", "#d55181"),  # 5 magenta
    ("#008300", "#008300"),  # 6 green
    ("#4a3aa7", "#9085e9"),  # 7 violet
    ("#e34948", "#e66767"),  # 8 red
]
CATEGORY_SLOT = {"none": None, "bo": 0, "llm": 1, "bo+llm": 2, "agentic": 6, "ceiling": None}
ACQ_SLOT = {"ei": 0, "ucb": 1, "pi": 2, "greedy": 3, "ts": 4}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def cat_of(arm: str) -> str:
    return CATEGORY.get(arm, "bo")


def cat_var(arm: str) -> str:
    slot = CATEGORY_SLOT[cat_of(arm)]
    return "var(--neutral)" if slot is None else f"var(--series-{slot + 1})"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _load_sweep(d: Path) -> dict | None:
    f = d / "summary.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    traces = []
    tf = d / "traces.jsonl"
    if tf.exists():
        for line in tf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(json.loads(line))
    data["_traces"] = traces
    return data


def load_chembl() -> dict | None:
    """The attribution ladder, with the agentic arms folded in when present.

    They come from two runs because they cost very differently -- the model arms
    are ~1,600 endpoint calls and the agentic controls are free -- but both are
    20 paired seeds of the identical library, budget and n_init, so the per-seed
    AUCs are comparable and the merge is legitimate. Where a run overlaps, the
    attribution run wins: it is the one whose arms were the question.
    """
    main = _load_sweep(CHEMBL_DIR)
    extra = _load_sweep(CHEMBL_AGENTIC_DIR)
    if main is None:
        return extra
    if extra is None:
        return main
    for name, arm in extra["arms"].items():
        main["arms"].setdefault(name, arm)
    for name, cmp in extra.get("comparisons_vs_reference", {}).items():
        main.setdefault("comparisons_vs_reference", {}).setdefault(name, cmp)
    main["_traces"] = main.get("_traces", []) + extra.get("_traces", [])
    return main


def load_pmo() -> dict | None:
    """Merge the 23-task control grid with the model-arm pass.

    The model arms could not run on all 23 tasks -- a per-round selector at
    budget 1000 and batch 10 is 99 calls per seed per task, and three such arms
    over 23 tasks is ~19 hours of endpoint time. They ran on a subset instead,
    and every table below reports which tasks each arm actually covers rather
    than averaging over a ragged grid as though it were complete.
    """
    base = json.loads(PMO_FILE.read_text(encoding="utf-8")) if PMO_FILE.exists() else None
    llm = json.loads(PMO_LLM_FILE.read_text(encoding="utf-8")) if PMO_LLM_FILE.exists() else None
    if base is None:
        return llm
    if llm is None:
        return base
    for task, per_arm in llm["results"].items():
        into = base["results"].setdefault(task, {})
        for arm, rows in per_arm.items():
            into.setdefault(arm, rows)
    base["_llm_tasks"] = sorted(llm["results"])
    merged_unavail = dict(base.get("unavailable_tasks") or {})
    merged_unavail.update(llm.get("unavailable_tasks") or {})
    base["unavailable_tasks"] = merged_unavail
    # `flat_tasks` is the runner's own record of a task that scored but did not
    # separate -- a real, hard task, not a broken oracle. Dropping it on the
    # merge left the report re-deriving "dead" from zero spread alone, which
    # merges the two causes it exists to keep apart.
    merged_flat = dict(base.get("flat_tasks") or {})
    merged_flat.update(llm.get("flat_tasks") or {})
    base["flat_tasks"] = merged_flat
    return base


def load_headroom() -> dict | None:
    if not HEADROOM.exists():
        return None
    return json.loads(HEADROOM.read_text(encoding="utf-8"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


# ---------------------------------------------------------------------------
# charts - inline SVG, no library
# ---------------------------------------------------------------------------
def bar_chart(rows, *, unit="AUC", width=760, row_h=30) -> str:
    """Horizontal bars with SEM whiskers, coloured by category.

    Horizontal because the category labels are long identifiers; rotating them
    under vertical bars is the single most common way a comparison becomes
    unreadable. Direct value labels on every bar, because three of the palette
    slots sit under 3:1 on the light surface and the relief rule applies.

    **The axis starts at zero.** A bar encodes a quantity by its length, so a
    baseline anywhere else multiplies every difference by an arbitrary factor:
    on this data the axis used to start at 0.53, which drew a 0.01 AUC gap as a
    third of a bar. If the arms are genuinely close the bars are meant to look
    close -- the numeric label on each bar carries the precision, and a
    difference that needs its own scale gets its own paired chart rather than a
    cropped axis.
    """
    rows = [r for r in rows if r.get("mean") is not None and not math.isnan(r["mean"])]
    if not rows:
        return '<p class="muted">nothing to plot</p>'

    label_w, right_w = 190, 86
    plot_w = width - label_w - right_w
    hi = max(r["mean"] + r.get("sem", 0) for r in rows)
    lo = min(0.0, min(r["mean"] - r.get("sem", 0) for r in rows))
    hi = hi + (hi - lo) * 0.06
    span = hi - lo or 1.0
    height = len(rows) * row_h + 34

    def x(v):
        return label_w + (v - lo) / span * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" width="100%" '
        f'style="max-width:{width}px" aria-label="{esc(unit)} by arm">'
    ]
    # recessive gridlines
    ticks = 5
    for i in range(ticks + 1):
        v = lo + span * i / ticks
        parts.append(
            f'<line x1="{x(v):.1f}" y1="14" x2="{x(v):.1f}" y2="{len(rows) * row_h + 16}" '
            f'class="grid"/>'
            f'<text x="{x(v):.1f}" y="{len(rows) * row_h + 30}" class="tick" '
            f'text-anchor="middle">{v:.2f}</text>'
        )

    for i, r in enumerate(rows):
        y = 18 + i * row_h
        bar_y = y + 5
        bh = row_h - 16
        w = max(1.0, x(r["mean"]) - label_w)
        colour = r.get("colour") or cat_var(r["name"])
        dash = ' stroke-dasharray="3 2" stroke-width="1.5" fill-opacity="0.22"' if cat_of(
            r["name"]
        ) == "ceiling" else ""
        parts.append(
            f'<text x="{label_w - 10}" y="{bar_y + bh - 2}" class="lbl" '
            f'text-anchor="end">{esc(r["name"])}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{bar_y}" width="{w:.1f}" height="{bh}" rx="4" '
            f'fill="{colour}" stroke="{colour}"{dash}>'
            f'<title>{esc(r["name"])}: {r["mean"]:.4f} +/- {r.get("sem", 0):.4f} '
            f'({esc(cat_of(r["name"]))})</title></rect>'
        )
        s = r.get("sem", 0)
        if s:
            x1, x2 = x(r["mean"] - s), x(r["mean"] + s)
            cy = bar_y + bh / 2
            parts.append(
                f'<line x1="{x1:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{cy:.1f}" class="whisk"/>'
                f'<line x1="{x1:.1f}" y1="{cy - 4:.1f}" x2="{x1:.1f}" y2="{cy + 4:.1f}" class="whisk"/>'
                f'<line x1="{x2:.1f}" y1="{cy - 4:.1f}" x2="{x2:.1f}" y2="{cy + 4:.1f}" class="whisk"/>'
            )
        parts.append(
            f'<text x="{width - right_w + 8}" y="{bar_y + bh - 2}" class="val">'
            f'{r["mean"]:.4f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def acquisition_timeline(log, *, width=760, h=26) -> str:
    """One cell per round, coloured by the acquisition the policy chose.

    This is the answer to "what was updated in the hyperparameters, per
    evaluation". An arm that switched once and an arm that switched every round
    can post the same AUC; only this strip separates them. Hover gives the round,
    the choice, xi, beta and the policy's stated reason.
    """
    if not log:
        return '<p class="muted">no policy log</p>'
    n = len(log)
    cw = max(2.0, width / n)
    parts = [
        f'<svg viewBox="0 0 {width} {h + 16}" role="img" width="100%" '
        f'style="max-width:{width}px" aria-label="acquisition per round">'
    ]
    for i, e in enumerate(log):
        d = e.get("decision", {})
        acq = d.get("acquisition", "?")
        slot = ACQ_SLOT.get(acq)
        colour = "var(--neutral)" if slot is None else f"var(--series-{slot + 1})"
        # 2px surface gap between adjacent fills, per the mark spec.
        parts.append(
            f'<rect x="{i * cw:.2f}" y="0" width="{max(1.0, cw - 2):.2f}" height="{h}" '
            f'rx="2" fill="{colour}">'
            f'<title>round {e.get("round", i)} (call {e.get("call", "?")}): {esc(acq)}'
            f'  xi={d.get("xi", "?")}  beta={d.get("beta", "?")}'
            f'\n{esc(d.get("rationale", ""))}</title></rect>'
        )
    parts.append(
        f'<text x="0" y="{h + 12}" class="tick">round 0</text>'
        f'<text x="{width}" y="{h + 12}" class="tick" text-anchor="end">round {n - 1}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def acq_legend() -> str:
    items = "".join(
        f'<span class="key"><i style="background:var(--series-{ACQ_SLOT[a] + 1})"></i>'
        f'<code>{esc(a)}</code> {esc(ACQ_LABEL[a])}</span>'
        for a in ACQ_ORDER
    )
    return f'<div class="legend">{items}</div>'


def cat_legend(present) -> str:
    items = []
    for c in CATEGORY_ORDER:
        if c not in present:
            continue
        slot = CATEGORY_SLOT[c]
        colour = "var(--neutral)" if slot is None else f"var(--series-{slot + 1})"
        items.append(
            f'<span class="key"><i style="background:{colour}"></i>'
            f'{CATEGORY_META[c][0]}</span>'
        )
    return f'<div class="legend">{"".join(items)}</div>'


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_ladder(chembl, pmo) -> str:
    ran = set()
    for src in (chembl, pmo):
        if not src:
            continue
        if "arms" in src:
            ran |= set(src["arms"])
        if "results" in src:
            for per_arm in src["results"].values():
                ran |= {k for k in per_arm if not k.startswith("_")}

    rows = []
    for c in CATEGORY_ORDER:
        # `LADDER_ARMS`, not `CATEGORY`: the latter also classifies arms that
        # only individual scripts build (`bo_maxmu_select`, `bo_clm_select`,
        # `bo_oracle_select`). Listing those here rendered them faded, which
        # this table's own legend reads as "this ladder arm did not run" --
        # about arms the ladder never contained. An arm outside the ladder is
        # still shown once it has actually run, so a merged run does not lose
        # a row.
        arms = [
            a
            for a, cc in CATEGORY.items()
            if cc == c and (a in LADDER_ARMS or a in ran)
        ]
        if not arms:
            continue
        title, why = CATEGORY_META[c]
        cells = []
        for a in sorted(arms):
            if a in ran:
                cells.append(f'<code class="ok">{esc(a)}</code>')
            elif a in NEEDS_MODEL:
                cells.append(f'<code class="skip">{esc(a)}</code>')
            else:
                cells.append(f'<code class="skip">{esc(a)}</code>')
        slot = CATEGORY_SLOT[c]
        colour = "var(--neutral)" if slot is None else f"var(--series-{slot + 1})"
        rows.append(
            f'<tr><td><span class="swatch" style="background:{colour}"></span>'
            f"<strong>{title}</strong></td>"
            f'<td>{" ".join(cells)}</td><td class="why">{why}</td></tr>'
        )

    return f"""
<section id="ladder">
  <h2>The ladder</h2>
  <p>Every arm exists to isolate one term of the same question &mdash;
  <strong>is it the BO, the LLM, or the two together?</strong> Arms in
  <code class="ok">solid</code> ran; <code class="skip">faded</code> ones did
  not, and the report says why rather than leaving a gap in a table.</p>
  <table class="ladder">
    <thead><tr><th>category</th><th>arms</th><th>what a win here would mean</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""


# The four contrasts the whole ladder exists to make. Each is (arm, control,
# what a positive difference would mean). Reading any one alone is how a project
# convinces itself an LLM helped: `bo_llm_select` beating `random` is compatible
# with the model contributing nothing and the GP carrying the arm.
CONTRASTS = [
    ("bo_ei", "random", "BO&rsquo;s own contribution",
     "A surrogate against a uniform draw, no model anywhere. Everything else is "
     "read against this: an LLM arm that merely matches it has bought nothing."),
    ("llm_only", "random", "the LLM&rsquo;s contribution, alone",
     "Identical uniform shortlist; the model picks instead of chance. No "
     "surrogate on either side, so this is the model&rsquo;s chemistry and "
     "nothing else."),
    ("bo_llm_select", "bo_random_select", "the LLM&rsquo;s contribution, on top of BO",
     "Identical GP shortlist; the model picks instead of chance. This is the "
     "only contrast that isolates what the model adds to a working optimiser."),
    ("bo_llm_select", "llm_only", "what BO adds underneath the model",
     "Same model, same prompt shape, same shortlist size &mdash; the shortlist "
     "comes from the GP rather than from a uniform draw."),
]


def _synthesise(stats: dict, arms: dict) -> str:
    """Say what the four rows mean together.

    Each row on its own is compatible with several readings; only the pair
    "the model helps alone" and "the model adds nothing on top of BO" pins the
    conclusion, and a reader should not have to derive it from four p-values.
    """
    alone = stats.get("llm_only_vs_random")
    on_top = stats.get("bo_llm_select_vs_bo_random_select")
    bo = stats.get("bo_ei_vs_random")
    if not (alone and on_top and bo):
        return ""

    bits = []
    if alone["p_value"] < 0.05 and alone["mean_diff"] > 0:
        ratio = bo["mean_diff"] / alone["mean_diff"] if alone["mean_diff"] else 0
        bits.append(
            f'With no surrogate underneath it the model beats a uniform draw '
            f'(<strong>{alone["mean_diff"]:+.4f}, p = {alone["p_value"]:.3f}</strong>), '
            f'so it is not picking at random &mdash; but the effect is '
            f'<strong>{ratio:.1f}&times; smaller than BO&rsquo;s</strong>.'
        )
    else:
        bits.append(
            'With no surrogate underneath it the model does not beat a uniform '
            'draw, so there is no evidence it reads the molecules at all.'
        )

    if on_top["p_value"] >= 0.05:
        bits.append(
            f'Once the GP is underneath it, the model adds nothing measurable '
            f'({on_top["mean_diff"]:+.4f}, p = {on_top["p_value"]:.3f}) &mdash; '
            f'whatever it knows, the surrogate already knew.'
        )
    else:
        bits.append(
            f'On top of BO it still moves the result '
            f'({on_top["mean_diff"]:+.4f}, p = {on_top["p_value"]:.3f}).'
        )

    # The baseline trap, stated with this run's own numbers.
    trap = ""
    if "bo_llm_select" in arms and "bo_ei" in arms:
        from molbo.metrics import compare_arms as _c
        vs_ei = _c(arms["bo_llm_select"]["aucs"], arms["bo_ei"]["aucs"], paired=True)
        if vs_ei["p_value"] < 0.05 <= on_top["p_value"]:
            trap = (
                f'<p class="warn"><strong>The baseline decides the headline.</strong> '
                f'Against <code>bo_ei</code>, <code>bo_llm_select</code> is a win: '
                f'{vs_ei["mean_diff"]:+.4f}, p = {vs_ei["p_value"]:.4f}. Against its '
                f'own control <code>bo_random_select</code> &mdash; the same GP '
                f'shortlist, the same size, a uniform pick instead of the '
                f'model&rsquo;s &mdash; the same arm gives '
                f'{on_top["mean_diff"]:+.4f}, p = {on_top["p_value"]:.4f}, ns. Same '
                f'arm, same data; one baseline says the model helps and the other '
                f'says it does not. The second is the honest one, because it is the '
                f'only one that holds the shortlist fixed.</p>'
            )

    return f'<p class="headline">{" ".join(bits)}</p>{trap}'


def section_attribution(chembl) -> str:
    """Is it the BO, the LLM, or the two together?"""
    if not chembl:
        return ""
    from molbo.metrics import compare_arms

    arms = chembl["arms"]

    # BO-vs-random is the yardstick every other row is read against.
    results: dict[str, float] = {}
    if "bo_ei" in arms and "random" in arms:
        results["bo_ei_vs_random"] = compare_arms(
            arms["bo_ei"]["aucs"], arms["random"]["aucs"], paired=True
        )["mean_diff"]

    rows = []
    stats: dict[str, dict] = {}
    for arm, ctrl, title, why in CONTRASTS:
        if arm not in arms or ctrl not in arms:
            rows.append(
                f'<tr><td><strong>{title}</strong><div class="why">{why}</div></td>'
                f'<td colspan="4" class="muted">needs <code>{esc(arm)}</code> and '
                f'<code>{esc(ctrl)}</code>; not both present</td></tr>'
            )
            continue
        a, b = arms[arm]["aucs"], arms[ctrl]["aucs"]
        if len(a) != len(b):
            rows.append(
                f'<tr><td><strong>{title}</strong></td>'
                f'<td colspan="4" class="muted">seed counts differ; not paired</td></tr>'
            )
            continue
        c = compare_arms(a, b, paired=True)
        stats[f"{arm}_vs_{ctrl}"] = c
        p_val = c["p_value"]
        stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                 else "*" if p_val < 0.05 else "ns")
        cls = "sig-up" if (c["mean_diff"] > 0 and p_val < 0.05) else (
            "sig-down" if (c["mean_diff"] < 0 and p_val < 0.05) else "")
        # The verdict has to carry magnitude, not just a sign and a star. The
        # LLM-alone effect is real (p = 0.008) and roughly a quarter the size of
        # BO's; calling both "carries the arm" would flatten exactly the
        # comparison this table exists to make.
        if p_val >= 0.05:
            verdict = "buys nothing measurable"
        elif c["mean_diff"] < 0:
            verdict = "actively hurts"
        else:
            ref = results.get("bo_ei_vs_random")
            share = (c["mean_diff"] / ref) if ref else None
            if share is None or share >= 0.75:
                verdict = "carries the arm"
            elif share >= 0.35:
                verdict = "helps, about half of BO&rsquo;s effect"
            else:
                verdict = f"helps, but {1 / share:.1f}&times; smaller than BO&rsquo;s"
        rows.append(
            f'<tr><td><strong>{title}</strong>'
            f'<div class="why">{why}</div>'
            f'<div class="pair"><code>{esc(arm)}</code> &minus; '
            f'<code>{esc(ctrl)}</code></div></td>'
            f'<td class="num {cls}">{c["mean_diff"]:+.4f}</td>'
            f'<td class="num">{c["cohens_dz"]:+.2f}</td>'
            f'<td class="num">{p_val:.4f} {stars}</td>'
            f'<td class="{cls}">{verdict}</td></tr>'
        )

    synthesis = _synthesise(stats, arms)

    return f"""
<section id="attribution">
  <h2>The question: BO, LLM, or the two together?</h2>
  <p>Four paired contrasts on CHEMBL203. Each isolates <em>one</em> term by
  holding everything else fixed &mdash; same library, same budget, same seed,
  same shortlist size &mdash; so the difference is the thing named and not the
  draw.</p>
  <table class="attr">
    <thead><tr><th>contrast</th><th>&Delta;AUC</th><th>d<sub>z</sub></th>
    <th>p</th><th>reading</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  {synthesis}
  <p class="muted">Wilcoxon signed-rank on per-seed AUC. &ldquo;Buys nothing
  measurable&rdquo; is not the same as &ldquo;is equal&rdquo;: with 20 seeds this
  design detects roughly a 0.02 AUC difference, so a real effect smaller than
  that would not show up here.</p>
</section>"""


def section_chembl(chembl) -> str:
    if not chembl:
        return (
            '<section id="chembl"><h2>Tier 1 &mdash; CHEMBL203 (EGFR)</h2>'
            '<p class="missing">Not run. Expected '
            "<code>results/chembl203_ladder_b50/summary.json</code>.</p></section>"
        )
    meta = chembl.get("meta", {})
    arms = chembl["arms"]
    rows = [
        {"name": n, "mean": s["auc_mean"], "sem": s["auc_sem"]}
        for n, s in arms.items()
    ]
    rows.sort(key=lambda r: -r["mean"])
    present = {cat_of(r["name"]) for r in rows}

    ref = chembl.get("reference", "bo_ei")
    cmp_rows = []
    for name, c in chembl.get("comparisons_vs_reference", {}).items():
        if "error" in c:
            continue
        p = c["p_value"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        cls = "sig-up" if (c["mean_diff"] > 0 and p < 0.05) else (
            "sig-down" if (c["mean_diff"] < 0 and p < 0.05) else ""
        )
        cmp_rows.append(
            f'<tr><td><code>{esc(name)}</code></td>'
            f'<td class="num {cls}">{c["mean_diff"]:+.4f}</td>'
            f'<td class="num">{c["cohens_dz"]:+.2f}</td>'
            f'<td class="num">{p:.4f}</td><td>{stars}</td></tr>'
        )

    # The ceiling is not eligible to be "the best arm": it reads the answer key.
    # Calling it that would report the reference line as a result.
    methods = [r for r in rows if cat_of(r["name"]) != "ceiling"]
    ceiling_row = next((r for r in rows if cat_of(r["name"]) == "ceiling"), None)
    # `CATEGORY.get`, not `cat_of`: the latter falls back to "bo" for an arm it
    # does not know, so a merged legacy sweep carrying e.g. `bo_llm_warmstart`
    # would be eligible to be named "best BO-only arm" -- an LLM arm reported as
    # the strongest result with no model in the loop.
    bo_only = [r for r in rows if CATEGORY.get(r["name"]) == "bo"]
    best_bo = max(bo_only, key=lambda r: r["mean"]) if bo_only else None
    bits = []
    if methods:
        best = max(methods, key=lambda r: r["mean"])
        bits.append(
            f'Best method: <code>{esc(best["name"])}</code> at '
            f'{best["mean"]:.4f} AUC'
        )
    if best_bo and (not methods or best_bo["name"] != best["name"]):
        bits.append(
            f'best BO-only arm <code>{esc(best_bo["name"])}</code> at '
            f'{best_bo["mean"]:.4f}'
        )
    if ceiling_row:
        bits.append(
            f'hindsight ceiling (<em>not</em> a method): '
            f'{ceiling_row["mean"]:.4f}'
        )
    headline = f'<p class="headline">{"; ".join(bits)}.</p>' if bits else ""

    return f"""
<section id="chembl">
  <h2>Tier 1 &mdash; CHEMBL203 (EGFR), whole target</h2>
  <p class="cfg">{esc(meta.get("library_size", "?"))} compounds &middot;
  budget {esc(meta.get("budget", "?"))} unique calls &middot;
  {len(meta.get("seeds", []))} paired seeds &middot;
  n_init {esc(meta.get("n_init", "?"))} &middot;
  best possible pIC50 {esc(round(meta.get("best_possible", 0), 2))}</p>
  {headline}
  {cat_legend(present)}
  {bar_chart(rows)}
  <h3>Paired comparison against <code>{esc(ref)}</code></h3>
  <p class="muted">Wilcoxon signed-rank on per-seed AUC, n={len(meta.get("seeds", []))}.
  Paired on the seed: both arms see the identical initial design, so the
  difference is the method rather than the draw.</p>
  <table class="stats">
    <thead><tr><th>arm</th><th>&Delta;AUC</th><th>d<sub>z</sub></th>
    <th>p</th><th></th></tr></thead>
    <tbody>{"".join(cmp_rows)}</tbody>
  </table>
</section>"""


def section_pmo(pmo) -> str:
    if not pmo:
        return (
            '<section id="pmo"><h2>Tier 2 &mdash; PMO, 23 tasks</h2>'
            '<p class="missing">Not run. Expected '
            "<code>results/pmo_ladder/pmo_results.json</code>.</p></section>"
        )
    results = pmo["results"]
    tasks = list(results.keys())
    arm_names: list[str] = []
    for per_arm in results.values():
        for k in per_arm:
            if not k.startswith("_") and k not in arm_names:
                arm_names.append(k)

    # Coverage is ragged on purpose: the control arms ran on all 23 tasks and
    # the model arms on a subset, because a per-round selector at budget 1000
    # is 99 calls per seed per task. A mean taken over each arm's own task list
    # would compare arms on different problems and is the easiest way to
    # manufacture a winner, so the aggregate below is restricted to the tasks
    # where *every* arm ran, and the coverage is stated on every row.
    # A task where every arm returns the identical score carries no comparison.
    # Three PMO oracles do this here: `drd2` and `gsk3b` load on a current
    # scikit-learn and then score every molecule identically (raw TDC puts
    # haloperidol, a D2 ligand, at 0.0), and `valsartan_smarts` matches nothing
    # in ZINC. Averaging them in drags every mean toward zero *and* compresses
    # the gaps between arms, so a dead task makes every method look more alike
    # than it is. They stay in the per-task table, flagged, and out of the
    # aggregates.
    def spread(t: str) -> float:
        vals = [
            mean([r["auc"] for r in results[t][a]])
            for a in arm_names if a in results[t]
        ]
        vals = [v for v in vals if not math.isnan(v)]
        return (max(vals) - min(vals)) if vals else 0.0

    dead = sorted(t for t in tasks if spread(t) < 1e-9)
    live = [t for t in tasks if t not in set(dead)]

    coverage = {a: sorted(t for t in live if a in results[t]) for a in arm_names}
    common = sorted(set(live).intersection(*(set(v) for v in coverage.values())))         if arm_names else []
    ragged = any(len(v) != len(live) for v in coverage.values())

    agg, wins = [], {a: 0 for a in arm_names}
    for t in common:
        per = results[t]
        scores = {a: mean([r["auc"] for r in per[a]]) for a in arm_names if a in per}
        if scores:
            wins[max(scores, key=scores.get)] += 1
    for a in arm_names:
        per_task = [mean([r["auc"] for r in results[t][a]]) for t in common]
        agg.append({"name": a, "mean": mean(per_task), "sem": sem(per_task)})
    agg.sort(key=lambda r: -r["mean"])
    present = {cat_of(r["name"]) for r in agg}

    # A second chart for the control arms alone, which do cover all 23 tasks.
    full_arms = [a for a in arm_names if len(coverage[a]) == len(live)]
    agg_full = []
    for a in full_arms:
        per_task = [mean([r["auc"] for r in results[t][a]]) for t in live]
        agg_full.append({"name": a, "mean": mean(per_task), "sem": sem(per_task)})
    agg_full.sort(key=lambda r: -r["mean"])

    dead_note = (
        '<p class="warn"><strong>'
        f'{len(dead)} task(s) return the identical score for every arm</strong> '
        + ", ".join(f"<code>{esc(t)}</code>" for t in dead)
        + '. They are excluded from every aggregate below and shown in the '
        'per-task table only. Two distinct causes, which a results file must '
        'not merge: <code>drd2</code> and <code>gsk3b</code> are <em>broken '
        'here</em> &mdash; their pickled scikit-learn models load on a current '
        'sklearn and then score every molecule identically (raw TDC scores '
        'haloperidol, a known D2 ligand, at 0.0) &mdash; while '
        '<code>valsartan_smarts</code> is <em>merely impossible</em>: its SMARTS '
        'matches nothing in ZINC-250k. Averaging either in would drag every mean '
        'toward zero and compress the gaps between arms, making every method '
        'look more alike than it is.</p>'
        if dead else ""
    )

    # Split the unavailable tasks by whether ANY run produced data for them.
    # `drd2` and `gsk3b` were skipped by the model-arm pass but had already been
    # swept by the control grid, which ran before the oracle probe existed --- so
    # they do appear below, as zeros. Saying they are "absent from every table"
    # would contradict the table three inches further down.
    unavail_all = pmo.get("unavailable_tasks") or {}
    unavail = {t: w for t, w in unavail_all.items() if t not in results}
    partial = {t: w for t, w in unavail_all.items() if t in results}

    unavail_note = (
        '<p class="warn"><strong>'
        f'{len(unavail)} task(s) produced no data at all.</strong> '
        + " ".join(
            f'<code>{esc(t)}</code> &mdash; {esc(why)}'
            for t, why in sorted(unavail.items())
        )
        + ' These are absent from every table below. A task that could not be '
        'scored is not a task where the arms tied, and the two must not look '
        'the same in a results file.</p>'
        if unavail else ""
    )
    if partial:
        unavail_note += (
            '<p class="warn"><strong>'
            f'{len(partial)} task(s) ran for the control arms but were refused '
            'for the model arms.</strong> '
            + " ".join(
                f'<code>{esc(t)}</code> &mdash; {esc(why)}'
                for t, why in sorted(partial.items())
            )
            + ' The control grid swept them before the oracle probe existed, so '
            'their rows below are real runs against a broken oracle &mdash; every '
            'arm scores 0.0000. They are excluded from every aggregate.</p>'
        )

    def _plural(n: int, word: str = "task") -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    coverage_note = (
        f'<p class="warn"><strong>Ragged grid.</strong> '
        f'{len(full_arms)} arm(s) ran on all {_plural(len(live))}; the model arms '
        f'ran on {min(len(v) for v in coverage.values())}. '
        f'The aggregate chart below is therefore restricted to the '
        f'<strong>{_plural(len(common))} where every arm ran</strong> &mdash; a '
        f'mean over each arm&rsquo;s own task list would compare arms on '
        f'different problems. The full-coverage chart beside it uses all '
        f'{_plural(len(live))} but only the arms that ran on all of them.</p>'
        if ragged else ""
    )

    # per-task table
    head = "".join(f"<th>{esc(a)}</th>" for a in arm_names)
    body = []
    for t in sorted(tasks):
        per = results[t]
        vals = {a: mean([r["auc"] for r in per[a]]) for a in arm_names if a in per}
        top = max(vals.values()) if vals else None
        cells = []
        for a in arm_names:
            if a not in vals or math.isnan(vals[a]):
                cells.append('<td class="num muted">&mdash;</td>')
                continue
            strong = top is not None and abs(vals[a] - top) < 1e-12
            cells.append(
                f'<td class="num{" best" if strong else ""}">{vals[a]:.3f}</td>'
            )
        if t in set(dead):
            note = (' <span class="flag" title="every arm scores identically; '
                    'excluded from all aggregates">not informative</span>')
        elif top is not None and (max(vals.values()) - min(vals.values())) < 0.005:
            note = ' <span class="flag" title="every arm within 0.005 AUC">flat</span>'
        else:
            note = ""
        body.append(f"<tr><th>{esc(t)}{note}</th>{''.join(cells)}</tr>")

    win_rows = "".join(
        f'<tr><td><code>{esc(a)}</code></td><td class="num">{wins[a]}</td>'
        f'<td class="num">{100 * wins[a] / max(1, len(common)):.0f}%</td>'
        # Denominator is `live`, not `tasks`: `coverage` is only ever counted
        # over the informative tasks, so dividing by the full task list printed
        # "19/22" for an arm that ran on every task it could and read as three
        # missing runs.
        f'<td class="num">{len(coverage[a])}/{len(live)}</td></tr>'
        for a in sorted(arm_names, key=lambda a: -wins[a])
    )

    return f"""
<section id="pmo">
  <h2>Tier 2 &mdash; PMO, {len(tasks)} tasks</h2>
  <p class="cfg">{esc(pmo.get("pool", "?"))} &middot; {esc(pmo.get("pool_size", "?")):}
  molecules &middot; budget {esc(pmo.get("budget", "?"))} unique calls &middot;
  {len(pmo.get("seeds", []))} seeds &middot; batch {esc(pmo.get("batch_size", "?"))}</p>
  <p class="warn"><strong>Not interchangeable with PMO's published tables.</strong>
  {esc(pmo.get("note", ""))} PMO's 25 published algorithms are generative
  (REINVENT, graph-GA, SMILES-LSTM and so on); the arms here all search a fixed
  enumerated ZINC pool, so the comparable published baselines are
  {", ".join("<code>" + esc(b) + "</code>" for b in pmo.get("comparable_pmo_baselines", []))}
  rather than the full leaderboard.</p>

  {coverage_note}
  {unavail_note}
  {dead_note}

  <h3>Every arm, on the {_plural(len(common))} they all share</h3>
  {cat_legend(present)}
  {bar_chart(agg)}

  <h3>Full coverage: the arms that ran on all {_plural(len(live))}</h3>
  <p class="muted">Same metric, every task, but only the arms with complete
  coverage. Read this for the BO-vs-random question and the chart above for
  anything involving a model arm.</p>
  {bar_chart(agg_full) if agg_full else '<p class="muted">no arm covers every task</p>'}

  <h3>Tasks won</h3>
  <p class="muted">A mean across tasks hides which tasks. This counts, over the
  {_plural(len(common))} shared, where each arm has the highest mean AUC. The last
  column is how many of the {_plural(len(live))} that arm ran on at all &mdash;
  &ldquo;did not run&rdquo; and &ldquo;ran and lost&rdquo; are different claims.</p>
  <table class="stats"><thead><tr><th>arm</th><th>tasks won</th><th>share</th>
  <th>coverage</th></tr></thead>
  <tbody>{win_rows}</tbody></table>

  <h3>Every task</h3>
  <p class="muted">Mean AUC over {len(pmo.get("seeds", []))} seeds. Bold is the
  best arm on that task. <span class="flag">flat</span> marks a task where every
  arm lands within 0.005 AUC &mdash; the task does not separate the methods, and
  reading a winner off it would be reading noise.</p>
  <div class="scroll">
  <table class="matrix"><thead><tr><th>task</th>{head}</tr></thead>
  <tbody>{"".join(body)}</tbody></table>
  </div>
</section>"""


def section_hyperparams(chembl, pmo) -> str:
    """The per-evaluation hyperparameter trace."""
    blocks = []

    # --- ChEMBL: traces.jsonl carries the full policy log per (arm, seed) ----
    if chembl and chembl.get("_traces"):
        by_arm: dict[str, list] = {}
        for t in chembl["_traces"]:
            if t.get("policy_log"):
                by_arm.setdefault(t["arm"], []).append(t)
        for arm in sorted(by_arm):
            tr = sorted(by_arm[arm], key=lambda t: t["seed"])[0]
            log = tr["policy_log"]
            counts: dict[str, int] = {}
            switches = 0
            prev = None
            betas = set()
            for e in log:
                a = e["decision"]["acquisition"]
                counts[a] = counts.get(a, 0) + 1
                if prev is not None and a != prev:
                    switches += 1
                prev = a
                betas.add(round(float(e["decision"].get("beta", 2.0)), 3))
            tally = " &middot; ".join(
                f'<code>{esc(a)}</code> {counts[a]}' for a in ACQ_ORDER if a in counts
            )
            blocks.append(
                f'<div class="trace"><h4><code>{esc(arm)}</code> '
                f'<span class="muted">CHEMBL203, seed {tr["seed"]}, '
                f"{len(log)} rounds, {switches} switches</span></h4>"
                f"{acquisition_timeline(log)}"
                f'<p class="tally">{tally} &middot; beta values used: '
                f'{", ".join(str(b) for b in sorted(betas))}</p></div>'
            )

    # --- PMO: one representative task -------------------------------------
    if pmo:
        for task, per in pmo["results"].items():
            keys = sorted(k for k in per if k.startswith("_policy_"))
            if not keys:
                continue
            for k in keys:
                arm = k[len("_policy_"):]
                seeds = per[k]
                if not seeds:
                    continue
                log = seeds[0]["log"]
                counts: dict[str, int] = {}
                for e in log:
                    a = e["decision"]["acquisition"]
                    counts[a] = counts.get(a, 0) + 1
                tally = " &middot; ".join(
                    f'<code>{esc(a)}</code> {counts[a]}' for a in ACQ_ORDER if a in counts
                )
                blocks.append(
                    f'<div class="trace"><h4><code>{esc(arm)}</code> '
                    f'<span class="muted">PMO {esc(task)}, seed '
                    f'{seeds[0]["seed"]}, {len(log)} rounds</span></h4>'
                    f"{acquisition_timeline(log)}"
                    f'<p class="tally">{tally}</p></div>'
                )
            break  # one task is enough to show the shape

    if not blocks:
        return (
            '<section id="hyper"><h2>Hyperparameters, per evaluation</h2>'
            '<p class="missing">No policy logs found. Only the '
            "<code>bo_policy_*</code> arms write one.</p></section>"
        )

    return f"""
<section id="hyper">
  <h2>Hyperparameters, per evaluation</h2>
  <p>Plain BO fixes the acquisition function and its hyperparameters
  <em>before</em> the run. The <code>bo_policy_*</code> arms re-choose them every
  round from the surrogate's own diagnostics. This is what changed, round by
  round &mdash; hover a cell for the round, the choice, <code>xi</code>,
  <code>beta</code> and the reason the policy gave.</p>
  <p class="muted">Two arms that switched a different number of times can post
  the same AUC. The summary tables cannot tell them apart; this can.</p>
  {acq_legend()}
  {"".join(blocks)}
</section>"""


def section_headroom(hr) -> str:
    if not hr:
        return ""
    by_budget: dict[int, list] = {}
    for r in hr["rows"]:
        by_budget.setdefault(r["budget"], []).append(r)
    blocks = []
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        fixed = [r for r in rows if r["policy"].startswith("fixed_")]
        # The instrument writes "ceiling (hindsight)", not "ceiling". Matching
        # on the exact string silently produced an empty table rather than an
        # error, which is the failure mode a report must never have.
        ceiling = next((r for r in rows if r["policy"].startswith("ceiling")), None)
        # `is_best_fixed` is computed by the instrument itself; prefer it over
        # recomputing, so the report cannot disagree with the run that made it.
        best_fixed = next(
            (r for r in fixed if r.get("is_best_fixed")),
            max(fixed, key=lambda r: r["auc"]) if fixed else None,
        )
        if not (best_fixed and ceiling):
            continue
        gap = ceiling["auc"] - best_fixed["auc"]
        inert = gap <= 2 * best_fixed["sem"]
        verdict = (
            "inside twice the SEM of the best fixed policy &mdash; "
            "<strong>the action space is inert here</strong>"
            if inert
            else "<strong>outside the noise</strong> &mdash; there is room to win"
        )
        blocks.append(
            f'<tr><td class="num">{budget}</td>'
            f'<td><code>{esc(best_fixed["policy"].replace("fixed_", ""))}</code> '
            f'{best_fixed["auc"]:.4f} &plusmn; {best_fixed["sem"]:.4f}</td>'
            f'<td class="num">{ceiling["auc"]:.4f}</td>'
            f'<td class="num {"sig-down" if inert else "sig-up"}">{gap:+.4f}</td>'
            f"<td>{verdict}</td></tr>"
        )

    return f"""
<section id="headroom">
  <h2>The gate: is the agentic action space winnable at all?</h2>
  <p>Run before spending a single model call. Every fixed acquisition alone,
  against a hindsight ceiling that switches each round to whichever acquisition
  reaches the best <em>true</em> value.</p>
  <p class="muted"><strong>Headroom is the ceiling minus the best fixed
  acquisition, not minus <code>ei</code>.</strong> If <code>ucb</code> alone beats
  <code>ei</code> alone, that is a better fixed policy, not evidence for
  adaptivity &mdash; and crediting it to an agent would repeat the mistake the
  warm-start control was built to catch.</p>
  <table class="stats">
    <thead><tr><th>budget</th><th>best fixed</th><th>ceiling</th>
    <th>headroom</th><th>verdict</th></tr></thead>
    <tbody>{"".join(blocks)}</tbody>
  </table>
  <p class="cfg">Library: <code>{esc(hr.get("dataset", "?"))}</code>,
  {esc(hr.get("library_size", "?"))} compounds, {len(hr.get("seeds", []))} seeds.</p>
</section>"""


def section_provenance(chembl, pmo) -> str:
    skipped = sorted(NEEDS_MODEL)
    ran = set()
    if chembl:
        ran |= set(chembl.get("arms", {}))
    if pmo:
        for per in pmo["results"].values():
            ran |= {k for k in per if not k.startswith("_")}
    not_run = [a for a in skipped if a not in ran]
    missing = (
        '<p class="warn"><strong>Model arms not in this report:</strong> '
        + ", ".join(f"<code>{esc(a)}</code>" for a in not_run)
        + ". They need a served model; the run was made with <code>--no-llm</code>, "
        "which records the skip rather than leaving a hole in the table. Every "
        "non-model arm below is complete.</p>"
        if not_run
        else '<p class="ok-note">Every arm in the ladder ran, model arms included.</p>'
    )

    # Counted, never written down. This sentence used to carry a hand-typed
    # "12-task PMO subset" while the run on disk covers nine, which is the
    # shape of error a generated results page exists to make impossible.
    model_tasks = 0
    if pmo:
        model_tasks = max(
            (
                sum(1 for per in pmo["results"].values() if a in per)
                for a in NEEDS_MODEL
            ),
            default=0,
        )
    subset = f"{model_tasks}-task PMO subset" if model_tasks else "a PMO subset"
    n_pmo_tasks = len(pmo["results"]) if pmo else 0
    return f"""
<section id="prov">
  <h2>Provenance</h2>
  {missing}
  <h3>Serving the model</h3>
  <p>Every model number below is <strong>Qwen3.5-4B, Q8_0 GGUF</strong> (unsloth),
  served locally by llama.cpp &mdash; the same model and quantisation as the
  project&rsquo;s earlier arms, so these results are directly comparable to them
  and no model-scale confound is introduced.</p>
  <pre>llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080 \
    --ctx-size 16384 --n-gpu-layers all --reasoning-budget 0</pre>
  <p class="muted">Greedy decoding (<code>temperature 0</code>,
  <code>top_k 1</code>, fixed seed) and llama.cpp compiles the JSON schema to a
  GBNF grammar, so a schema-valid response is guaranteed by construction. Every
  response is cached under a hash of the exact request <em>including</em> the
  backend&rsquo;s sampling settings, so a rerun makes zero model calls.</p>

  <h4>A model that was tried and not used</h4>
  <p class="muted">IBM Granite 4.2 8B (Q4_K_M) was served first, as the
  larger-model control. It works, but it needs the <em>opposite</em> reasoning
  flags to Qwen, and the recipe this repo documents for Qwen actively breaks it.
  Recorded because the failure is silent in one case and cryptic in the other:</p>
  <table class="stats">
    <thead><tr><th>server flags</th><th>result on granite</th></tr></thead>
    <tbody>
      <tr><td><em>none</em></td>
      <td>empty content, <code>finish_reason=length</code> &mdash; the model spent
      its whole budget thinking and never emitted the JSON</td></tr>
      <tr><td><code>--reasoning off --reasoning-budget 0</code><br>
      <span class="muted">(the documented Qwen recipe)</span></td>
      <td>HTTP 500, <code>Unexpected empty grammar stack</code> on
      <strong>every</strong> schema &mdash; budget 0 forces a think block open, the
      reply begins with <code>&lt;/think&gt;</code>, and the JSON grammar applies
      from token 0, so that tag can never be produced</td></tr>
      <tr><td><code>--reasoning off --reasoning-format deepseek</code></td>
      <td>works &mdash; the thought goes to <code>message.reasoning_content</code>
      and <code>message.content</code> is bare JSON</td></tr>
    </tbody>
  </table>
  <p class="muted">Granite was dropped in favour of Qwen because Qwen is the
  model every other arm in this project was run on, so keeping it removes a
  model-scale confound rather than introducing one. Throughput was <em>not</em>
  the reason: measured end to end through the campaign loop the two are
  comparable &mdash; Qwen3.5-4B Q8_0 sustains about <strong>38 schema-constrained
  calls per minute</strong> against granite 8B Q4_K_M&rsquo;s <strong>~40</strong>.
  That rate is what caps the model arms at a {subset}: a per-round
  selector is 99 calls per seed per task, so two such arms across all
  {n_pmo_tasks} tasks would be roughly 10 hours of decode.</p>

  <h3>Commands</h3>
  <pre># the gate - no endpoint, no GPU. Run this first.
python scripts/acquisition_headroom.py \
    --dataset data/chembl/egfr_anilinoquinazoline.csv \
    --out results/acquisition_headroom.json

# tier 1a: the attribution ladder - BO vs LLM vs BO+LLM, 20 paired seeds
python scripts/run_lookup_experiment.py \
    --dataset data/chembl/egfr_all.csv --ladder \
    --llm-backend llamacpp --llm-model qwen3.5-4b \
    --budget 50 --seeds 20 \
    --arms random bo_ei llm_only bo_random_select bo_llm_select \
    --out results/chembl203_ladder_b50

# tier 1b: the agentic arms and the hindsight ceiling (no endpoint needed)
python scripts/run_lookup_experiment.py \
    --dataset data/chembl/egfr_all.csv --ladder --no-llm --ceiling \
    --budget 50 --seeds 20 --out results/chembl203_agentic_b50

# tier 2a: all 23 PMO tasks, control arms only
python scripts/run_pmo_experiment.py --ladder --no-llm \
    --seeds 5 --budget 1000 --tasks &lt;all 23&gt; --out results/pmo_ladder

# tier 2b: the model arms, all 23 tasks. --policy-every 10 is the cost dial:
# bo_policy_llm would otherwise ask once per round, 99 times per seed per task.
python scripts/run_pmo_experiment.py --ladder \
    --llm-backend llamacpp --llm-model qwen3.5-4b \
    --policy-every 10 --seeds 5 --budget 1000 \
    --tasks &lt;all 23&gt; --out results/pmo_ladder_llm

# this page
python scripts/build_ladder_report.py</pre>
  <p class="muted">Generated {esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}
  from whatever was in <code>results/</code>. Nothing here is hand-entered.</p>
</section>"""


# ---------------------------------------------------------------------------
CSS = """
:root{
  color-scheme:light;
  --bg:#fbfbfa; --surface:#ffffff; --border:#e4e3df;
  --text:#14140f; --text-2:#52514e; --muted:#7c7b76; --neutral:#9b9a93;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#eda100;
  --series-5:#e87ba4; --series-6:#008300; --series-7:#4a3aa7; --series-8:#e34948;
  --up:#1a7f4f; --down:#b3261e; --warn-bg:#fdf6e6; --warn-bd:#e8d49a;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#131312; --surface:#1a1a19; --border:#33322e;
    --text:#ffffff; --text-2:#c3c2b7; --muted:#96958c; --neutral:#7e7d76;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
    --series-5:#d55181; --series-6:#008300; --series-7:#9085e9; --series-8:#e66767;
    --up:#5fbf8b; --down:#f2a099; --warn-bg:#2a2412; --warn-bd:#5c4d22;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#131312; --surface:#1a1a19; --border:#33322e;
  --text:#ffffff; --text-2:#c3c2b7; --muted:#96958c; --neutral:#7e7d76;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
  --series-5:#d55181; --series-6:#008300; --series-7:#9085e9; --series-8:#e66767;
  --up:#5fbf8b; --down:#f2a099; --warn-bg:#2a2412; --warn-bd:#5c4d22;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:48px 16px 96px}
header h1{font-size:30px;line-height:1.25;margin:0 0 8px;letter-spacing:-.02em}
.lede{color:var(--text-2);font-size:17px;margin:0 0 6px;max-width:64ch}
section{margin:56px 0 0}
h2{font-size:21px;margin:0 0 10px;letter-spacing:-.01em;
  padding-bottom:8px;border-bottom:1px solid var(--border)}
h3{font-size:16px;margin:28px 0 8px}
h4{font-size:14px;margin:0 0 6px;font-weight:600}
p{margin:0 0 12px;max-width:74ch}
code{font:12.5px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1px 5px}
pre{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;overflow-x:auto;font:12.5px/1.65 ui-monospace,Menlo,Consolas,monospace;
  color:var(--text-2)}
pre code{border:0;background:none;padding:0}
.muted{color:var(--muted);font-size:13.5px}
.cfg{color:var(--text-2);font-size:13.5px}
.headline{font-size:16px;font-weight:600}
.missing{color:var(--muted);font-style:italic}
.warn{background:var(--warn-bg);border:1px solid var(--warn-bd);
  border-radius:8px;padding:10px 14px;font-size:13.5px;max-width:none}
.ok-note{color:var(--up);font-size:13.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 4px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);
  vertical-align:top}
thead th{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);font-weight:600;border-bottom:1px solid var(--border)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,Consolas,monospace}
td.best{font-weight:700;color:var(--text)}
.sig-up{color:var(--up)} .sig-down{color:var(--down)}
.why{color:var(--text-2);font-size:13px}
.pair{font-size:12px;color:var(--muted);margin-top:4px}
.attr td:first-child{max-width:42ch}
.attr tbody tr{border-bottom:1px solid var(--border)}
.swatch{display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:8px;vertical-align:baseline}
.ladder code.ok{border-color:var(--text-2)}
.ladder code.skip{opacity:.42}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 0 14px;
  font-size:12.5px;color:var(--text-2)}
.key{display:inline-flex;align-items:center;gap:7px}
.key i{width:11px;height:11px;border-radius:3px;display:inline-block}
svg{display:block;margin:4px 0 10px;overflow:visible}
.grid{stroke:var(--border);stroke-width:1}
.tick{fill:var(--muted);font-size:10.5px;
  font-family:ui-monospace,Menlo,Consolas,monospace}
.lbl{fill:var(--text-2);font-size:12px;
  font-family:ui-monospace,Menlo,Consolas,monospace}
.val{fill:var(--text);font-size:12px;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,Consolas,monospace}
.whisk{stroke:var(--text-2);stroke-width:1.5;opacity:.75}
.trace{margin:18px 0 22px;padding:14px 16px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px}
.tally{font-size:12.5px;color:var(--text-2);margin:2px 0 0}
.flag{font-size:10.5px;color:var(--muted);border:1px solid var(--border);
  border-radius:3px;padding:0 4px;vertical-align:middle}
.scroll{overflow-x:auto;border:1px solid var(--border);border-radius:8px}
.matrix{margin:0;font-size:12.5px}
.matrix th:first-child{position:sticky;left:0;background:var(--bg);
  font-weight:500;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.matrix thead th{white-space:nowrap;font-size:10.5px}
@media (max-width:640px){
  .wrap{padding:28px 16px 64px} header h1{font-size:24px} .lede{font-size:15.5px}
}
"""


def build() -> Path:
    chembl, pmo, hr = load_chembl(), load_pmo(), load_headroom()

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BO vs LLM vs BO+LLM</title>
<style>{CSS}</style>
<body>
<div class="wrap">
<header>
  <h1>Is it the BO, the LLM, or the two together?</h1>
  <p class="lede">One arm ladder, run on two tiers: CHEMBL203 (EGFR, whole
  target) and all 23 PMO tasks over ZINC-250k. Every arm isolates one term, and
  each has the control it has to beat.</p>
</header>
{section_ladder(chembl, pmo)}
{section_attribution(chembl)}
{section_headroom(hr)}
{section_chembl(chembl)}
{section_pmo(pmo)}
{section_hyperparams(chembl, pmo)}
{section_provenance(chembl, pmo)}
</div>
</body>
</html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


def check(page: str) -> list[str]:
    """Structural problems that would otherwise ship silently.

    An empty table renders as a heading with nothing under it, which reads as
    "there was no effect" rather than "the report failed to find the data". That
    is the one failure mode a results page must not have, and it is exactly what
    happened when this file matched the headroom ceiling on the wrong string.
    """
    import re

    problems = []
    for i, body in enumerate(re.findall(r"<tbody>(.*?)</tbody>", page, re.S)):
        if not body.strip():
            problems.append(f"table {i + 1} rendered with an empty <tbody>")
    if re.search(r"nan", page):
        problems.append("a NaN reached the page")
    if "{" in re.sub(r"<style>.*?</style>", "", page, flags=re.S):
        problems.append("an unsubstituted f-string brace reached the page")
    return problems


def main() -> int:
    out = build()
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")

    problems = check(out.read_text(encoding="utf-8"))
    if problems:
        print("")
        print("PROBLEMS:")
        for why in problems:
            print(f"  ! {why}")
    else:
        print("  structure checks passed")
    for name, path in (
        ("ChEMBL203", CHEMBL_DIR / "summary.json"),
        ("PMO", PMO_FILE),
        ("headroom", HEADROOM),
    ):
        print(f"  {name:10s} {'found' if path.exists() else 'MISSING'}  {path}")
    # Non-zero, so a scripted or CI build of the report cannot pass while the
    # page it wrote has an empty table in it. Printing alone made `check` an
    # advisory note, which is the same silence it was written to remove.
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

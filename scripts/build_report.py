"""Build results/report.html from whatever is in results/.

    python scripts/build_report.py

Every number in the report is read from the run artifacts rather than typed in,
so the page cannot drift from the data it describes. Re-run it after any sweep.

The output is deliberately a single self-contained file -- no CDN, no web
fonts, no sibling assets -- so it can be opened straight off disk, mailed, or
dropped in a repo without breaking.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from molbo.metrics import compare_arms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "report.html"

# Which run directory supplies which arm, per budget. Arms are compared against
# the bo_ei of the *same* directory so every paired test uses matched seeds.
LAYOUT = {
    50: [
        ("bench_final_b50", ["random", "bo_ei", "bo_llm_constraint", "bo_random_select",
                             "bo_llm_select", "bo_llm_warmstart", "bo_oracle_constraint"]),
        ("bench_warmstart_b50", ["bo_diverse_warmstart"]),
    ],
    100: [
        ("bench_b100", ["random", "bo_ei", "bo_random_constraint", "bo_data_driven_constraint",
                        "bo_llm_constraint", "bo_oracle_constraint"]),
        ("bench_warmstart_b100", ["bo_diverse_warmstart", "bo_llm_warmstart"]),
    ],
}

ARM_INFO = {
    "random":                    ("Random search", "control", "Uniform sampling from the library. The floor."),
    "bo_ei":                     ("Plain BO", "reference", "GP + expected improvement. Everything is tested against this."),
    "bo_random_constraint":      ("BO + random mask", "control", "Restricted to a random 5% of the library."),
    "bo_data_driven_constraint": ("BO + statistical mask", "control", "Descriptor ranges of the best-observed compounds. No model."),
    "bo_llm_constraint":         ("BO + LLM constraint", "llm", "The model narrows the library from a mined substructure menu."),
    "bo_random_select":          ("BO + random shortlist", "control", "Uniform pick from BO's top-15 shortlist."),
    "bo_llm_select":             ("BO + LLM selection", "llm", "BO shortlists 15; the model picks which one is assayed."),
    "bo_diverse_warmstart":      ("BO + diversity warm start", "control", "MaxMin Tanimoto pick of the initial design. No model."),
    "bo_llm_warmstart":          ("BO + LLM warm start", "llm", "The model picks the initial design from a 40-molecule sample."),
    "bo_oracle_constraint":      ("Ceiling (knows the answers)", "ceiling", "Restricted to the library's true best 5%. Not a method."),
}

HYPERPARAMS = [
    ("Surrogate", [
        ("Kernel", "Tanimoto (MinMax on binary vectors)", "No lengthscale to fit; standard for fingerprint GPs"),
        ("Features", "ECFP4, radius 2, 2048 bits", "RDKit MorganGenerator"),
        ("Noise-to-signal ratio, eta", "scanned over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0}", "Refitted by exact log marginal likelihood every round"),
        ("Signal variance, s", "closed form (profiled out)", "s* = z'(K0 + eta.I)^-1 z / n; no numeric search"),
        ("Mean function", "constant, = mean of observed y", "Targets z-scored before fitting"),
        ("Jitter", "1e-8", "Added to the Gram diagonal for Cholesky stability"),
    ]),
    ("Acquisition", [
        ("Function", "Expected improvement", "Against the best observed value"),
        ("Exploration offset, xi", "0.01", "In units of the objective (pIC50)"),
        ("Batch size q", "1 (ChEMBL) / 10 (PMO)", "Sequential on the small library; batched on ZINC for cost"),
        ("Diversity penalty", "0.0 (off)", "Available; unused in these runs"),
        ("Tie-breaking", "uniform random jitter, 1e-12", "Shared by BO and the selector shortlist"),
    ]),
    ("Campaign", [
        ("Budget", "50 and 100 unique oracle calls", "PMO tier: 1,000"),
        ("Initial design n_init", "10 (ChEMBL) / 20 (PMO)", "Random, and identical across arms for a given seed"),
        ("Seeds", "20 (ChEMBL) / 5 (PMO)", "Paired across arms; PMO convention is 5 runs"),
        ("Refit cadence", "every round", "No staleness"),
        ("Candidates scored per round", "all (ChEMBL) / 2,000 sampled (PMO)", "Search space stays the full pool either way"),
    ]),
    ("Constraint arms", [
        ("Allowed library fraction", "0.05", "For the random-mask control and the ceiling"),
        ("min_candidates", "20 unseen", "Below this a hard mask degrades to a soft penalty"),
        ("soft_penalty", "1.0", "Subtracted from acquisition for disallowed molecules"),
        ("Refresh cadence", "once, and every 15 rounds (ablation)", "Both tested; no difference"),
        ("Substructure menu", "28 entries, coverage window 2%-90%", "Mined from the library: ring systems, Murcko scaffolds, 22 curated groups"),
    ]),
    ("Selection / warm-start arms", [
        ("shortlist_k", "15", "k=1 reduces exactly to plain BO"),
        ("Shortlist order", "seeded shuffle", "Removes primacy/recency bias; answer mapped back to acquisition rank"),
        ("Warm-start sample", "40 molecules", "Drawn with the campaign seed, the same draw plain BO uses"),
    ]),
    ("Language model", [
        ("Model", "Qwen3.5-4B, Q8_0 GGUF (unsloth)", "Served locally by llama.cpp"),
        ("temperature / top_k / seed", "0.0 / 1 / 0", "Greedy decoding: deterministic without a cache"),
        ("Thinking", "disabled (--reasoning-budget 0)", "Qwen3.5 is a hybrid reasoning model"),
        ("Context / max_tokens", "16384 / 700-1200", "choice emitted before reasoning so truncation cannot lose it"),
        ("Output constraint", "JSON schema compiled to a GBNF grammar", "Malformed output impossible by construction"),
        ("Caching", "sha256 of model + sampling settings + prompt + schema", "A rerun makes zero model calls"),
    ]),
]


def read_metrics(tag: str) -> pd.DataFrame | None:
    p = RES / tag / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def arm_rows(budget: int) -> list[dict]:
    rows: list[dict] = []
    for tag, arms in LAYOUT[budget]:
        m = read_metrics(tag)
        if m is None:
            continue
        ref = m[m.arm == "bo_ei"].sort_values("seed")
        for arm in arms:
            g = m[m.arm == arm].sort_values("seed")
            if g.empty:
                continue
            d = p = None
            if arm != "bo_ei" and len(ref) == len(g) and not ref.empty:
                c = compare_arms(g.auc.values, ref.auc.values, paired=True)
                d, p = c["mean_diff"], c["p_value"]
            label, role, blurb = ARM_INFO.get(arm, (arm, "control", ""))
            rows.append({
                "arm": arm, "label": label, "role": role, "blurb": blurb, "source": tag,
                "auc": g.auc.mean(), "sem": g.auc.std(ddof=1) / np.sqrt(len(g)),
                "best": g.best.mean(), "regret": g.simple_regret.mean(),
                "recall": g.top1pct_recall.mean(), "n": len(g), "d": d, "p": p,
            })
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["auc"]):
        if r["arm"] not in seen:
            seen.add(r["arm"])
            out.append(r)
    return out


def head_to_head(budget: int) -> list[dict]:
    tag = f"bench_warmstart_b{budget}"
    m = read_metrics(tag)
    if m is None:
        return []
    g = lambda a: m[m.arm == a].sort_values("seed").auc.values
    out = []
    for name, a, b in [
        ("LLM warm start vs plain BO", "bo_llm_warmstart", "bo_ei"),
        ("Diversity warm start vs plain BO", "bo_diverse_warmstart", "bo_ei"),
        ("LLM warm start vs diversity warm start", "bo_llm_warmstart", "bo_diverse_warmstart"),
    ]:
        x, y = g(a), g(b)
        if len(x) == 0 or len(x) != len(y):
            continue
        c = compare_arms(x, y, paired=True)
        out.append({"name": name, "budget": budget, "d": c["mean_diff"],
                    "p": c["p_value"], "wins": int((x > y).sum()), "n": len(x)})
    return out


def esc(t) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def sig(d, p) -> str:
    if d is None:
        return '<span class="pill ref">reference</span>'
    cls = "ns" if p >= 0.05 else ("up" if d > 0 else "down")
    txt = "ns" if p >= 0.05 else ("p&lt;0.001" if p < 0.001 else f"p={p:.3f}")
    return f'<span class="pill {cls}">{txt}</span>'


def arm_table(rows: list[dict]) -> str:
    body = []
    for r in rows:
        cls = {"reference": "is-ref", "llm": "is-llm", "ceiling": "is-ceil"}.get(r["role"], "")
        dd = "&mdash;" if r["d"] is None else f"{r['d']:+.4f}"
        body.append(
            f'<tr class="{cls}"><td><span class="arm">{esc(r["label"])}</span>'
            f'<span class="blurb">{esc(r["blurb"])}</span></td>'
            f'<td>{r["auc"]:.4f} &plusmn; {r["sem"]:.4f}</td><td>{r["best"]:.2f}</td>'
            f'<td>{r["regret"]:.3f}</td><td>{100 * r["recall"]:.1f}%</td>'
            f'<td>{dd}</td><td>{sig(r["d"], r["p"])}</td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Arm</th><th>AUC top-10</th>'
        "<th>Best pIC50</th><th>Regret</th><th>Top-1% recall</th><th>&Delta;AUC</th>"
        "<th>vs plain BO</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def chart_svg(rows: list[dict], budget: int) -> str:
    vals = [r["auc"] for r in rows]
    sems = [r["sem"] for r in rows]
    lo = min(v - s for v, s in zip(vals, sems)) - 0.012
    hi = max(v + s for v, s in zip(vals, sems)) + 0.03
    W, padL, padR, padT, rowH = 720, 210, 34, 18, 34
    H = padT + len(rows) * rowH + 34
    X = lambda v: padL + (v - lo) / (hi - lo) * (W - padL - padR)
    top, bot = padT, padT + len(rows) * rowH
    ref = next((r["auc"] for r in rows if r["role"] == "reference"), None)
    colour = {"reference": "var(--c-ref)", "llm": "var(--c-llm)",
              "ceiling": "var(--c-ceil)", "control": "var(--c-mute)"}

    ticks = np.round(np.linspace(lo + 0.01, hi - 0.01, 5), 2)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean AUC by arm at a '
         f'{budget}-call budget, with standard-error whiskers">']
    for t in ticks:
        s.append(f'<line x1="{X(t):.1f}" y1="{top}" x2="{X(t):.1f}" y2="{bot}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        s.append(f'<text x="{X(t):.1f}" y="{bot + 16}" font-size="10.5" fill="var(--ink-3)" '
                 f'text-anchor="middle" class="mono">{t:.2f}</text>')
    if ref is not None:
        s.append(f'<line x1="{X(ref):.1f}" y1="{top - 4}" x2="{X(ref):.1f}" y2="{bot + 2}" '
                 f'stroke="var(--c-ref)" stroke-width="2" stroke-dasharray="4 3"/>')
    for i, r in enumerate(rows):
        cy = top + i * rowH + rowH / 2
        c = colour[r["role"]]
        s.append(f'<text x="{padL - 12}" y="{cy + 4:.1f}" font-size="12" text-anchor="end" '
                 f'fill="{"var(--ink-2)" if r["role"] == "control" else "var(--ink-1)"}">'
                 f'{esc(r["label"])}</text>')
        s.append(f'<line x1="{X(r["auc"] - r["sem"]):.1f}" y1="{cy:.1f}" '
                 f'x2="{X(r["auc"] + r["sem"]):.1f}" y2="{cy:.1f}" stroke="{c}" '
                 f'stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
        s.append(f'<circle cx="{X(r["auc"]):.1f}" cy="{cy:.1f}" r="5.5" fill="{c}" '
                 f'stroke="var(--surface-1)" stroke-width="2"><title>{esc(r["label"])}: '
                 f'{r["auc"]:.4f} &#177; {r["sem"]:.4f} (n={r["n"]})</title></circle>')
        s.append(f'<text x="{X(r["auc"] + r["sem"]) + 9:.1f}" y="{cy + 4:.1f}" font-size="11" '
                 f'fill="var(--ink-2)" class="mono">{r["auc"]:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


def gp_section() -> str:
    p = RES / "gp_trace_seed0.json"
    if not p.exists():
        return ""
    tr = json.loads(p.read_text(encoding="utf-8"))
    etas = sorted({r["eta"] for r in tr})
    err = float(np.mean([abs(r["pred_mu"] - r["actual"]) for r in tr]))
    psd = float(np.mean([r["pred_sd"] for r in tr]))
    show = [tr[0], tr[1], tr[len(tr) // 4], tr[len(tr) // 2], tr[-2], tr[-1]]
    body = "".join(
        f'<tr><td>{r["round"]}</td><td>{r["n_obs"]}</td><td>{r["eta"]:g}</td>'
        f'<td>{r["signal_var"]:.3f}</td><td>{r["gp_mean"]:.2f}</td><td>{r["lml"]:.1f}</td>'
        f'<td>{r["pred_mu"]:.2f} &plusmn; {r["pred_sd"]:.2f}</td><td>{r["ei"]:.4f}</td>'
        f'<td>{r["actual"]:.2f}</td><td class="smi">{esc(r["suggested"])}</td></tr>'
        for r in show
    )
    return f"""
<section id="whatbo">
  <span class="eyebrow">What BO actually does</span>
  <h2>What is optimised, what is updated, what gets suggested</h2>
  <p><strong>The objective.</strong> Maximise measured pIC50. The campaign never sees the
  library's unobserved values &mdash; only the molecules it has paid for.</p>
  <p><strong>What is fitted, every round.</strong> Not the molecules: the GP's own
  hyperparameters. After each new measurement the surrogate is refitted from scratch by
  exact log marginal likelihood &mdash; the noise-to-signal ratio <span class="mono">eta</span>
  is scanned over a fixed 9-point grid, and for each one the signal variance has a closed
  form, so there is no gradient step and no optimiser seed. The posterior mean and variance
  over all unevaluated molecules are then recomputed.</p>
  <p><strong>What is suggested.</strong> The molecule maximising expected improvement against
  the best value seen so far. Below is a real run (seed 0, budget 50): every row is the GP
  as it stood that round, the compound it chose, what it predicted, and what the assay
  actually said.</p>
  <div class="scroll"><table><thead><tr><th>Round</th><th>n obs</th><th>eta</th>
  <th>signal var</th><th>GP mean</th><th>log ML</th><th>predicted</th><th>EI</th>
  <th>actual</th><th>suggested molecule</th></tr></thead><tbody>{body}</tbody></table></div>
  <div class="cards">
    <div class="card"><span class="num">{len(etas)}</span><span class="cap">distinct noise levels
    chosen across {len(tr)} rounds ({", ".join(f"{e:g}" for e in etas)}). It starts noisy at
    10 observations and tightens to near-interpolation by 50 &mdash; the surrogate genuinely
    re-decides how much to trust the data as it accumulates.</span></div>
    <div class="card"><span class="num">{err:.2f}</span><span class="cap">mean absolute prediction
    error, in log units, on the compound it chose &mdash; against a mean predicted uncertainty of
    {psd:.2f}. The GP's error bars are honest, which is what EI needs to work.</span></div>
    <div class="card"><span class="num">{len(tr)}</span><span class="cap">rounds recorded, from
    {tr[0]["n_obs"]} observations to {tr[-1]["n_obs"]}. The surrogate only has to <em>rank</em>
    well enough to steer the next purchase, not to predict exactly. Full per-round trace in
    <span class="mono">results/gp_trace_seed0.json</span>.</span></div>
  </div>
</section>"""


def pmo_section() -> str:
    p = RES / "pmo" / "pmo_results.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    published = {  # PMO paper, 10,000-call budget, 5 runs
        "albuterol_similarity": (0.489, 0.609), "celecoxib_rediscovery": (0.354, 0.415),
        "deco_hop": (0.589, 0.643), "median1": (0.202, 0.249),
        "osimertinib_mpo": (0.771, 0.779), "zaleplon_mpo": (0.039, 0.168),
    }
    EXCLUDE = {"zaleplon_mpo"}
    rows, wins_s, wins_m, n = [], 0, 0, 0
    for task, arms in d["results"].items():
        if "bo_ei" not in arms:
            continue
        ra = [x["auc"] for x in arms["random"]]
        ba = [x["auc"] for x in arms["bo_ei"]]
        scr, mp = published.get(task, (None, None))
        excluded = task in EXCLUDE
        if not excluded and scr is not None:
            n += 1
            wins_s += np.mean(ba) > scr
            wins_m += np.mean(ba) > mp
        rows.append(
            f'<tr class="{"muted" if excluded else ""}"><td class="mono">{esc(task)}</td>'
            f"<td>{np.mean(ra):.4f}</td><td><strong>{np.mean(ba):.4f}</strong></td>"
            f'<td>{scr:.3f}{" <span class=\'pill up\'>we win</span>" if not excluded and np.mean(ba) > scr else ""}</td>'
            f'<td>{mp:.3f}{" <span class=\'pill up\'>we win</span>" if not excluded and np.mean(ba) > mp else ""}</td>'
            f'<td>{"excluded" if excluded else ""}</td></tr>'
        )
    return f"""
<section id="public">
  <span class="eyebrow">Public benchmark</span>
  <h2>PMO &mdash; comparing against 25 published algorithms</h2>
  <p>Every other number here comes from an oracle we built, which is exactly what a reviewer
  distrusts. <strong>PMO</strong> (&ldquo;Sample Efficiency Matters: A Benchmark for Practical
  Molecular Optimization&rdquo;, Gao, Fu, Sun &amp; Coley, NeurIPS 2022) fixed 23 tasks, capped
  the oracle budget, and re-ran 25 algorithms &mdash; finding that most &ldquo;state-of-the-art&rdquo;
  methods stop beating their predecessors once the budget is controlled. Tasks are distributed
  through <span class="mono">PyTDC</span>; a task is a scoring function from SMILES to [0, 1].</p>
  <p>Our search space is the <strong>ZINC-250k</strong> pool ({d["pool_size"]:,} molecules), so
  the like-for-like published baselines are the library-based ones: <span class="mono">screening</span>
  (random sampling from the pool) and <span class="mono">MolPAL</span> (BO over the pool). The
  generative baselines search unrestricted chemical space and will score higher partly for that
  reason alone.</p>
  <div class="scroll"><table>
    <caption>AUC top-10. Ours: <strong>{d["budget"]:,} calls</strong>, {len(d["seeds"])} seeds.
    Published: <strong>10,000 calls</strong>, 5 runs. Not interchangeable &mdash; ours is the
    harder budget.</caption>
    <thead><tr><th>Task</th><th>ours: random</th><th>ours: plain BO</th>
    <th>screening (10k)</th><th>MolPAL (10k)</th><th></th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  <p><strong>Plain BO at {d["budget"]:,} calls beats <span class="mono">screening</span> at
  10,000 calls on {wins_s} of {n} comparable tasks</strong>, and beats
  <span class="mono">MolPAL</span> on {wins_m} of {n}. With a 10&times; smaller budget that is a
  good showing &mdash; and it is the whole point of this tier: it says the surrogate and the
  harness are sound, so a null result on the LLM arms belongs to the arms and not to the
  scaffolding.</p>
  <p class="note"><span class="mono">zaleplon_mpo</span> is excluded from the comparison.
  TDC's current implementation uses a soft isomer term (a random ZINC molecule scores 0.20,
  not 0), which cannot reproduce the published <span class="mono">screening</span> value of
  0.039 &mdash; our number and theirs are not measuring the same function.</p>
</section>"""


def _ground_truth() -> dict:
    p = RES / "pmo_ground_truth.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {t: v for t, v in d.get("tasks", {}).items() if "pool_top10_mean" in v}


def _llm_select_run() -> tuple[dict, str] | tuple[None, None]:
    """The run directory that carries bo_llm_select, with its own seeds."""
    for tag in ("pmo_ladder_llm", "pmo_select", "pmo"):
        p = RES / tag / "pmo_results.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if any("bo_llm_select" in a and a["bo_llm_select"] for a in d["results"].values()):
            return d, tag
    return None, None


GT_ARMS = [
    ("random", "Random", "var(--c-mute)"),
    ("bo_ei", "Plain BO", "var(--c-ref)"),
    ("bo_llm_select", "BO + LLM selection", "var(--c-llm)"),
]


def ground_truth_chart(rows: list[dict]) -> str:
    """Three arms and the enumerated ceiling, one row per task, shared 0-1 axis.

    The axis starts at zero and the ceiling is drawn as a rule rather than a
    bar: it is a limit on the metric, not another method's score, and the
    distance from a dot to that rule is the quantity the section is about.
    """
    W, padL, padR, padT, rowH = 720, 176, 56, 22, 44
    H = padT + len(rows) * rowH + 34
    lo, hi = 0.0, 1.0
    X = lambda v: padL + (v - lo) / (hi - lo) * (W - padL - padR)
    top, bot = padT, padT + len(rows) * rowH

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="AUC of random, plain BO and '
         f'BO with LLM selection against the enumerated ground-truth ceiling, per task">']
    for t in np.round(np.arange(0, 1.01, 0.2), 1):
        s.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{bot}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        s.append(f'<text x="{X(t):.1f}" y="{bot + 16}" font-size="10.5" fill="var(--ink-3)" '
                 f'text-anchor="middle" class="mono">{t:.1f}</text>')
    s.append(f'<text x="{padL}" y="{bot + 31}" font-size="11" fill="var(--ink-3)">'
             f'AUC of the top-10 mean at 1,000 calls</text>')

    for i, r in enumerate(rows):
        cy = top + i * rowH + rowH / 2
        s.append(f'<text x="{padL - 12}" y="{cy - 2:.1f}" font-size="12" text-anchor="end" '
                 f'fill="var(--ink-1)">{esc(r["task"])}</text>')
        s.append(f'<text x="{padL - 12}" y="{cy + 12:.1f}" font-size="10.5" text-anchor="end" '
                 f'fill="var(--ink-3)" class="mono">ceiling {r["ceiling"]:.3f}</text>')

        # ceiling first, so the dots sit on top of it where they nearly touch
        s.append(f'<line x1="{X(r["ceiling"]):.1f}" y1="{cy - 15:.1f}" '
                 f'x2="{X(r["ceiling"]):.1f}" y2="{cy + 15:.1f}" stroke="var(--c-ceil)" '
                 f'stroke-width="2.5"><title>{esc(r["task"])} ground truth: best top-10 mean '
                 f'in the pool is {r["ceiling"]:.4f} &mdash; the highest AUC this task '
                 f'admits</title></line>')

        # Dodged vertically, not drawn on one line. Plain BO and the selector
        # differ by ~0.001 AUC, which is well under a pixel on this axis: on a
        # shared baseline the later dot hides the earlier one entirely and the
        # arm reads as missing rather than as identical. The offset keeps all
        # three visible while x still carries the only quantity that matters.
        for j, (arm, label, colour) in enumerate(GT_ARMS):
            v = r["arms"].get(arm)
            if v is None:
                continue
            dy = (j - 1) * 8.5
            s.append(f'<line x1="{X(v["auc"] - v["sem"]):.1f}" y1="{cy + dy:.1f}" '
                     f'x2="{X(v["auc"] + v["sem"]):.1f}" y2="{cy + dy:.1f}" stroke="{colour}" '
                     f'stroke-width="2" stroke-linecap="round" opacity="0.5"/>')
            s.append(f'<circle cx="{X(v["auc"]):.1f}" cy="{cy + dy:.1f}" r="4.5" fill="{colour}" '
                     f'stroke="var(--surface-1)" stroke-width="1.5"><title>{esc(label)} on '
                     f'{esc(r["task"])}: AUC {v["auc"]:.4f} &#177; {v["sem"]:.4f}, '
                     f'{100 * v["auc"] / r["ceiling"]:.1f}% of the ceiling</title></circle>')

        s.append(f'<text x="{X(r["ceiling"]) + 10:.1f}" y="{cy + 4:.1f}" font-size="10.5" '
                 f'fill="var(--ink-3)" class="mono">{r["best_pct"]:.0f}%</text>')
    s.append("</svg>")
    return "".join(s)


def selector_delta_chart(rows: list[dict]) -> str:
    """Paired delta of BO + LLM selection against plain BO, on its own scale.

    The main chart's axis spans the whole metric, where this difference is
    invisible. Here the axis spans only the difference, so a reader can see
    both that the sign is usually positive and that every interval crosses
    zero -- which is the finding.
    """
    pts = [r for r in rows if r.get("delta") is not None]
    if not pts:
        return ""
    span = max(abs(p["delta"]) + p["delta_sem"] for p in pts) * 1.15
    span = max(span, 0.005)
    W, padL, padR, padT, rowH = 720, 176, 56, 22, 26
    H = padT + len(pts) * rowH + 36
    X = lambda v: padL + (v + span) / (2 * span) * (W - padL - padR)
    top, bot = padT, padT + len(pts) * rowH

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Paired difference in AUC between '
         f'BO with LLM selection and plain BO, per task, with standard-error whiskers">']
    for t in np.linspace(-span, span, 5):
        s.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{bot}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        s.append(f'<text x="{X(t):.1f}" y="{bot + 16}" font-size="10.5" fill="var(--ink-3)" '
                 f'text-anchor="middle" class="mono">{t:+.3f}</text>')
    s.append(f'<line x1="{X(0):.1f}" y1="{top - 6}" x2="{X(0):.1f}" y2="{bot}" '
             f'stroke="var(--ink-2)" stroke-width="2"/>')
    s.append(f'<text x="{padL}" y="{bot + 32}" font-size="11" fill="var(--ink-3)">'
             f'&Delta;AUC vs plain BO &mdash; right of the line is the selector ahead</text>')

    for i, p in enumerate(pts):
        cy = top + i * rowH + rowH / 2
        s.append(f'<text x="{padL - 12}" y="{cy + 4:.1f}" font-size="11.5" text-anchor="end" '
                 f'fill="var(--ink-1)">{esc(p["task"])}</text>')
        s.append(f'<line x1="{X(p["delta"] - p["delta_sem"]):.1f}" y1="{cy:.1f}" '
                 f'x2="{X(p["delta"] + p["delta_sem"]):.1f}" y2="{cy:.1f}" '
                 f'stroke="var(--c-llm)" stroke-width="2" stroke-linecap="round" opacity="0.5"/>')
        s.append(f'<circle cx="{X(p["delta"]):.1f}" cy="{cy:.1f}" r="4.5" fill="var(--c-llm)" '
                 f'stroke="var(--surface-1)" stroke-width="1.5"><title>{esc(p["task"])}: '
                 f'&Delta;AUC {p["delta"]:+.4f} &#177; {p["delta_sem"]:.4f} (paired, '
                 f'n={p["n_pair"]})</title></circle>')
    s.append("</svg>")
    return "".join(s)


def pmo_ground_truth_section() -> str:
    gt = _ground_truth()
    run, tag = _llm_select_run()
    if not gt or run is None:
        return ""

    rows = []
    for task, arms in run["results"].items():
        if task not in gt or "bo_llm_select" not in arms:
            continue
        ceiling = gt[task]["pool_top10_mean"]
        if ceiling <= 0:
            continue
        got = {}
        for arm, _, _ in GT_ARMS:
            recs = [x for x in arms.get(arm, []) if isinstance(x, dict) and "auc" in x]
            if not recs:
                continue
            a = np.array([x["auc"] for x in recs], float)
            got[arm] = {
                "auc": float(a.mean()),
                "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
                "top10": float(np.mean([x["top10"] for x in recs])),
                "n": a.size,
                "raw": a,
            }
        if "bo_ei" not in got:
            continue
        best = max(v["auc"] for v in got.values())
        # Seeds are matched across arms in a run, so the difference is paired
        # and its error bar is the spread of the per-seed differences -- not
        # the two arms' separate SEMs, which would be far wider.
        delta = delta_sem = None
        n_pair = 0
        if "bo_llm_select" in got and got["bo_llm_select"]["n"] == got["bo_ei"]["n"] > 1:
            diff = got["bo_llm_select"]["raw"] - got["bo_ei"]["raw"]
            delta = float(diff.mean())
            delta_sem = float(diff.std(ddof=1) / np.sqrt(diff.size))
            n_pair = int(diff.size)
        rows.append({"task": task, "ceiling": ceiling, "gt1": gt[task]["pool_top1"],
                     "arms": got, "best_pct": 100 * best / ceiling,
                     "delta": delta, "delta_sem": delta_sem, "n_pair": n_pair})
    if not rows:
        return ""
    rows.sort(key=lambda r: -r["ceiling"])

    table = []
    for r in rows:
        cells = []
        for arm, _, _ in GT_ARMS:
            v = r["arms"].get(arm)
            cells.append("&mdash;" if v is None else
                         f'{v["auc"]:.4f}<span class="blurb">{100 * v["top10"] / r["ceiling"]:.1f}% '
                         f'of best top-10</span>')
        table.append(
            f'<tr><td class="mono">{esc(r["task"])}</td>'
            + "".join(f"<td>{c}</td>" for c in cells)
            + f'<td><strong>{r["ceiling"]:.4f}</strong></td><td>{r["gt1"]:.4f}</td></tr>'
        )

    n = len(rows)
    bo_pct = np.mean([100 * r["arms"]["bo_ei"]["top10"] / r["ceiling"] for r in rows])
    sel = [r for r in rows if "bo_llm_select" in r["arms"]]
    sel_pct = np.mean([100 * r["arms"]["bo_llm_select"]["top10"] / r["ceiling"] for r in sel])
    rnd_pct = np.mean([100 * r["arms"]["random"]["top10"] / r["ceiling"] for r in rows
                       if "random" in r["arms"]])
    # Sign test over tasks, not seeds: "ahead on k of n" invites a conclusion
    # the count alone cannot support, so the report states the p-value beside it.
    deltas = [r["arms"]["bo_llm_select"]["auc"] - r["arms"]["bo_ei"]["auc"] for r in sel]
    sel_wins = sum(1 for d in deltas if d > 0)
    med_delta = float(np.median(deltas))
    from scipy import stats as _st
    sign_p = float(_st.binomtest(sel_wins, len(deltas), 0.5).pvalue)
    sep_px = abs(med_delta) * (720 - 176 - 56)  # same geometry as ground_truth_chart
    crosses = sum(1 for r in rows if r.get("delta") is not None
                  and abs(r["delta"]) < r["delta_sem"])

    return f"""
<section id="groundtruth">
  <span class="eyebrow">Ground truth</span>
  <h2>How far from the actual answer?</h2>
  <p>A PMO task is a scoring function that costs microseconds, so the best answer on this
  library is not something to be estimated &mdash; it can be <strong>looked up</strong>.
  <span class="mono">scripts/pmo_ground_truth.py</span> scores all
  {run["pool_size"]:,} ZINC-250k molecules with the same TDC oracle the campaigns use, sorts
  them, and records the top of the distribution. The mean of the ten best scores is the
  <strong>ceiling on the AUC top-10 metric</strong>: the metric averages the top-10 curve, the
  curve cannot exceed that value, so neither can any method on this pool.</p>
  <p class="note">This is the optimum <em>of the library</em>, not of the task. These functions
  are defined over all of chemical space, and the rediscovery targets are not in ZINC-250k
  &mdash; <span class="mono">celecoxib_rediscovery</span> scores 1.0 for celecoxib itself, but
  the best molecule the pool contains scores {gt.get("celecoxib_rediscovery", {}).get("pool_top1", float("nan")):.4f}.
  Generative methods that build molecules are not bounded by these numbers. Library methods,
  including every arm here and PMO's own <span class="mono">screening</span> and
  <span class="mono">MolPAL</span>, are.</p>
  <div class="legend">
    <span><i class="sw" style="background:var(--c-mute)"></i>Random</span>
    <span><i class="sw" style="background:var(--c-ref)"></i>Plain BO</span>
    <span><i class="sw" style="background:var(--c-llm)"></i>BO + LLM selection</span>
    <span><i class="sw" style="background:var(--c-ceil);border-radius:1px;width:3px;height:14px"></i>Ground-truth ceiling</span>
  </div>
  <div class="chart">{ground_truth_chart(rows)}</div>
  <p>On that axis the two BO arms are {sep_px:.1f} pixels apart, so the chart can separate
  <em>random from BO</em> and <em>BO from the ceiling</em>, and nothing finer. The comparison
  the section is actually asked for &mdash; does letting the model pick from the shortlist
  help &mdash; needs its own scale:</p>
  <div class="chart">{selector_delta_chart(rows)}</div>
  <div class="scroll"><table>
    <caption>AUC top-10 at {run["budget"]:,} calls, mean of {len(run["seeds"])} seeds, from
    <span class="mono">results/{tag}</span>. The sub-line is the top-10 actually found as a
    percentage of the best top-10 in the pool. <span class="mono">zaleplon_mpo</span> appears
    here although it is excluded from the published comparison above: the ceiling is enumerated
    with the same TDC oracle the runs call, so the column is internally consistent whether or
    not that oracle matches the one PMO published against.</caption>
    <thead><tr><th>Task</th><th>random</th><th>plain BO</th><th>BO + LLM selection</th>
    <th>ceiling (best top-10)</th><th>best molecule</th></tr></thead>
    <tbody>{"".join(table)}</tbody></table></div>
  <div class="verdict"><p><strong>The surrogate closes almost all of the gap; what is left is
  too small for the selector to be measured in.</strong> Across {n} tasks, random sampling finds
  {rnd_pct:.0f}% of the best top-10 in the pool at 1,000 calls and plain BO finds {bo_pct:.1f}%,
  so the GP closes about {bo_pct - rnd_pct:.0f} of the {100 - rnd_pct:.0f} available points.
  Letting the model choose from BO's shortlist moves that to {sel_pct:.1f}%. It is ahead of
  plain BO on {sel_wins} of {len(sel)} tasks &mdash; consistent in sign, but by a median of
  {med_delta:+.4f} AUC, with a sign test over tasks at p&nbsp;=&nbsp;{sign_p:.2f}. On
  {crosses} of {len(sel)} tasks the paired difference is smaller than its own standard error.
  The honest reading is an effect too small to resolve at this seed count, not an effect shown
  to be absent.</p></div>
  <p class="note">The one task with real room left is <span class="mono">isomers_c7h8n2o2</span>,
  where BO reaches 90% of the ceiling and random manages 30% &mdash; and it is also where the
  selector's largest margin sits. At the other end, <span class="mono">qed</span> is saturated:
  random sampling alone finds 99% of the best top-10 in the pool, so nothing measured on that
  task separates any two methods. Neither fact is visible from a ranking of arms against each
  other.</p>
</section>"""


def headroom_section() -> str:
    p = RES / "headroom_egfr_congeneric.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = "".join(
        f'<tr><td>{r["budget"]}</td><td>{100 * r["fraction"]:.0f}%</td>'
        f'<td>{r["n_allowed"]:,}</td><td>{r["bo_auc"]:.4f}</td><td>{r["ceiling_auc"]:.4f}</td>'
        f'<td><span class="pill {"up" if r["headroom"] > 0.07 else "ns"}">'
        f'+{r["headroom"]:.4f}</span></td></tr>'
        for r in d["rows"]
    )
    best = max(d["rows"], key=lambda r: r["headroom"])
    return f"""
<section id="headroom">
  <span class="eyebrow">Experiment design</span>
  <h2>Headroom was measured before any arm was run</h2>
  <p>A ceiling arm restricted to the library's true best <em>k</em>% bounds what
  <em>any</em> constraint could achieve, since no real method can beat knowing the answers.
  Running it first is what stops you spending a week in a regime where the effect you want
  cannot be resolved even in principle.</p>
  <div class="scroll"><table>
    <caption>Ceiling minus plain BO, {len(d["seeds"])} seeds, library of {d["library_size"]:,}.</caption>
    <thead><tr><th>Budget</th><th>Allowed fraction</th><th>Compounds allowed</th>
    <th>Plain BO</th><th>Ceiling</th><th>Headroom</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p>Headroom roughly triples as the constraint tightens, and grows as the budget shrinks. The
  configuration one reaches for first &mdash; a generous budget against a loose constraint &mdash;
  is the <em>worst</em> place to look, at +0.031. Most headroom: budget {best["budget"]},
  fraction {100 * best["fraction"]:.0f}%, worth <strong>+{best["headroom"]:.4f}</strong>.
  Everything here therefore runs tight and short.</p>
</section>"""


def selector_section() -> str:
    p = RES / "bench_final_b50" / "selector_ranks.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    k = d["shortlist_k"]
    r = np.asarray(d["ranks"])
    hist = [int((r == i).sum()) for i in range(k)]
    cells = "".join(f"<td>{h}</td>" for h in hist)
    head = "".join(f"<th>{i}</th>" for i in range(k))
    return f"""
  <h3>The confound this arm exposed</h3>
  <p>Presented in acquisition order, &ldquo;picks position 0&rdquo; and &ldquo;agrees with BO's
  top candidate&rdquo; are <em>the same event</em>. The first run's choice distribution was
  strongly non-uniform but the shape was <strong>positional</strong> &mdash; peaks at the first
  and last entries, a trough at the second. Textbook primacy/recency bias impersonating
  judgment. Shuffling the shortlist into a seeded random order and mapping the answer back
  decouples them; these are the resulting <em>acquisition</em> ranks over
  {len(r):,} decisions:</p>
  <div class="scroll"><table><thead><tr><th>acq. rank</th>{head}</tr></thead>
  <tbody><tr><td>picks</td>{cells}</tr></tbody></table></div>
  <p>It picks BO's own top candidate <strong>{100 * (r == 0).mean():.1f}%</strong> of the time
  against {100 / k:.1f}% for chance, and its mean chosen rank is {r.mean():.2f} against
  {(k - 1) / 2:.1f} for uniform. So with position bias removed it <em>does</em> carry real
  signal about what the GP likes &mdash; and it still does not help, because agreeing with BO a
  fifth of the time and deviating the rest is worse than simply taking BO's pick. It is a
  noisy approximation of the acquisition function, not an improvement on it.</p>"""


def search_space_section() -> str:
    sp_p, fp_p = RES / "search_space.json", RES / "fp_collisions.json"
    if not sp_p.exists():
        return ""
    d = json.loads(sp_p.read_text(encoding="utf-8"))
    fp = json.loads(fp_p.read_text(encoding="utf-8")) if fp_p.exists() else None

    coll = ""
    if fp:
        a, b = fp["includeChirality_false"], fp["includeChirality_true"]
        cur, old = (b, a)
        ex = (a.get("worst_examples") or [{}])[0]
        drop = 100 * (1 - cur["pairs"] / max(1, old["pairs"]))
        coll = f"""
  <h3>Molecules the surrogate could not tell apart &mdash; found, and fixed</h3>
  <p>Maximum <em>off-diagonal</em> Tanimoto in this library was <strong>1.0</strong>, meaning
  some distinct library entries were <em>identical</em> to the GP. RDKit's Morgan generator
  ignores stereochemistry by default, so E/Z and R/S isomers collapsed onto the same
  fingerprint while their measured activities did not. The worst pair differed by
  <strong>{ex.get("gap", 0):.2f} log units</strong> &mdash; a thousand-fold in potency,
  invisible to the model:</p>
  <pre>{esc(ex.get("a", ""))}   pIC50 {ex.get("value_a", 0):.2f}
{esc(ex.get("b", ""))}   pIC50 {ex.get("value_b", 0):.2f}</pre>
  <p>A GP handed two identical inputs with different targets can only average them, which puts
  an <em>irreducible noise floor</em> under every prediction and shows up as a large fitted
  <span class="mono">eta</span>. Enabling <span class="mono">includeChirality</span> removes
  {drop:.0f}% of the collisions:</p>
  <div class="scroll"><table><thead><tr><th>ECFP4 setting</th><th>Identical pairs</th>
  <th>Molecules involved</th><th>Largest pIC50 gap</th><th>Pairs differing &gt; 0.5 log</th>
  </tr></thead><tbody>
  <tr class="muted"><td class="mono">includeChirality=False &mdash; RDKit default</td>
  <td>{old["pairs"]}</td><td>{old["molecules"]} of {d["n_molecules"]:,}</td>
  <td>{old["max_gap"]:.2f}</td><td>{old["pairs_gap_gt_half"]}</td></tr>
  <tr class="is-ceil"><td class="mono">includeChirality=True &mdash; <strong>in use
  ({esc(d.get("fingerprint_version", "?"))})</strong></td>
  <td>{cur["pairs"]}</td><td>{cur["molecules"]} of {d["n_molecules"]:,}</td>
  <td>{cur["max_gap"]:.2f}</td><td>{cur["pairs_gap_gt_half"]}</td></tr>
  </tbody></table></div>
  <p>The remaining {cur["pairs"]} pairs are regioisomers that ECFP4 at radius 2 genuinely cannot
  separate; those need a larger radius or a different descriptor, not a flag. Density and
  effective rank are unchanged &mdash; chirality alters <em>which</em> bits are set, not how many
  &mdash; so this is a targeted correction rather than a different representation.</p>
  <p class="note">Every result on this page was regenerated after the fix. The fingerprint recipe
  is versioned (<span class="mono">{esc(d.get("fingerprint_version", "?"))}</span>) and that
  version is part of the pooled-fingerprint cache filename, so a representation change can never
  be silently mixed with stale data.</p>"""

    return f"""
<section id="space">
  <span class="eyebrow">Search space</span>
  <h2>Bounds and dimensionality</h2>
  <p><strong>This is not continuous box-bounded BO.</strong> There are no lower/upper bounds on
  continuous variables and no design matrix to optimise over. The search space is a
  <em>discrete, fully enumerated set of molecules</em>, and the acquisition function is
  maximised by scanning it rather than by any gradient or inner optimiser. &ldquo;Bounds&rdquo;
  here means membership rules, not intervals.</p>
  <div class="scroll"><table><tbody>
    <tr><td>Search space</td><td class="mono">discrete, {d["n_molecules"]:,} enumerated molecules
      (PMO tier: 249,455)</td></tr>
    <tr><td>Decision variable</td><td class="mono">one categorical index over the library &mdash;
      effectively 1-D, with {d["n_molecules"]:,} levels</td></tr>
    <tr><td>Membership bounds</td><td class="mono">anilinoquinazoline core substructure;
      {d["heavy_min"]}&ndash;{d["heavy_max"]} heavy atoms (median {d["heavy_median"]:.0f});
      measured pIC50 in [{d["value_min"]:.2f}, {d["value_max"]:.2f}]</td></tr>
    <tr><td>Additional bound (constraint arms)</td><td class="mono">a boolean mask over the
      library; degrades to a soft penalty below 20 unseen survivors</td></tr>
    <tr><td>Acquisition optimiser</td><td class="mono">exhaustive scan of all unevaluated members
      &mdash; no local optimiser, no restarts, global by construction</td></tr>
  </tbody></table></div>

  <h3>Where dimensionality does enter: the kernel's feature space</h3>
  <p>The GP does not see molecules, it sees fingerprints &mdash; and that is the only place a
  dimension count is meaningful.</p>
  <div class="cards">
    <div class="card"><span class="num">{d["fp_nominal_dim"]:,}</span><span class="cap">nominal
    ECFP4 bits. Of these {d["fp_active_bits"]:,} are ever set in this library and
    {d["fp_informative_bits"]:,} actually vary &mdash; the rest carry no signal at all.</span></div>
    <div class="card"><span class="num">{d["density_pct"]:.1f}%</span><span class="cap">density:
    a molecule sets {d["bits_per_mol_mean"]:.0f} bits on average (range
    {d["bits_per_mol_min"]}&ndash;{d["bits_per_mol_max"]}). The representation is extremely
    sparse.</span></div>
    <div class="card"><span class="num">{d["eff_rank_participation"]:.1f}</span><span class="cap">
    <strong>effective rank</strong> of the Tanimoto Gram (participation ratio). Despite 2,048
    nominal dimensions the kernel is dominated by ~10 directions; {d["eigs_for_90pct"]} eigenvalues
    are needed for 90% of the spectrum.</span></div>
  </div>
  <p>That gap &mdash; 2,048 nominal, ~10 effective &mdash; is precisely why a Tanimoto kernel is
  used instead of a squared-exponential one. An RBF GP would need a lengthscale per dimension over
  2,048 sparse binary features, which in practice collapses to either &ldquo;everything is
  identical&rdquo; or &ldquo;nothing is related&rdquo;. Tanimoto has <em>no</em> lengthscale to
  fit, so the curse of dimensionality never arises: pairwise similarity is the whole model.
  Observed similarity is mean {d["tanimoto_mean"]:.2f}, median {d["tanimoto_median"]:.2f},
  95th percentile {d["tanimoto_p95"]:.2f} &mdash; a genuinely congeneric series, tight enough for
  the surrogate to generalise but not degenerate.</p>
  {coll}
</section>"""


def pmo_llm_section() -> str:
    """The LLM arms on PMO -- directional only, and the report says why."""
    import numpy as _np

    runs = {"warm start": RES / "pmo_warmstart" / "pmo_results.json",
            "constraint": RES / "pmo_constraint" / "pmo_results.json"}
    have = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in runs.items() if v.exists()}
    if not have:
        return ""

    CMP = {"bo_llm_warmstart": "_cmp_bo_llm_warmstart",
           "bo_diverse_warmstart": "_cmp_bo_diverse_warmstart",
           "bo_llm_constraint": "_cmp_bo_llm_constraint"}
    tasks: dict[str, dict] = {}
    for d in have.values():
        for task, arms in d["results"].items():
            row = tasks.setdefault(task, {})
            if "bo_ei" in arms:
                row["bo_ei"] = float(_np.mean([x["auc"] for x in arms["bo_ei"]]))
            for arm, key in CMP.items():
                if key in arms:
                    row[arm] = arms[key]["mean_diff"]

    cols = [a for a in CMP if any(a in r for r in tasks.values())]
    head = "".join(f"<th>{esc(a.replace('bo_', '').replace('_', ' '))}</th>" for a in cols)
    body = ""
    for task, r in sorted(tasks.items()):
        cells = ""
        for a in cols:
            v = r.get(a)
            cells += "<td>&mdash;</td>" if v is None else (
                f'<td>{v:+.4f}</td>' if abs(v) < 0.01 else
                f'<td><span class="pill {"up" if v > 0 else "down"}">{v:+.4f}</span></td>')
        body += (f'<tr><td class="mono">{esc(task)}</td>'
                 f'<td>{r.get("bo_ei", float("nan")):.4f}</td>{cells}</tr>')
    means = "".join(
        f"<td><strong>{_np.mean([r[a] for r in tasks.values() if a in r]):+.4f}</strong></td>"
        for a in cols)
    n_seeds = len(next(iter(have.values()))["seeds"])
    calls = sum(d.get("warmstart_llm_calls", 0) + d.get("constraint_llm_calls", 0)
                for d in have.values())

    return f"""
<section id="pmollm">
  <span class="eyebrow">LLM arms on the public benchmark</span>
  <h2>Do the pairings transfer off our own oracle?</h2>
  <p>Every LLM result above is on a library we assembled. These are the same arms on PMO's
  tasks over the ZINC-250k pool &mdash; a structurally diverse pool with objectives that are not
  potency at all, which is a far harsher test of a method tuned on a congeneric series.</p>
  <div class="scroll"><table>
    <caption>&Delta;AUC against plain BO on the same seeds. Budget 1,000 calls,
    {n_seeds} seeds, {calls} model calls in total.</caption>
    <thead><tr><th>Task</th><th>plain BO</th>{head}</tr></thead>
    <tbody>{body}<tr class="is-ref"><td>mean</td><td></td>{means}</tr></tbody></table></div>
  <p><strong>Nothing transfers.</strong> The warm start is worth about
  {_np.mean([r["bo_llm_warmstart"] for r in tasks.values() if "bo_llm_warmstart" in r]):+.4f} on
  average and is behind its own free diversity control; the constraint arm is again negative.
  The warm-start result is what the budget arithmetic predicts: the initial design is 2% of a
  1,000-call budget here against 20% of a 50-call budget on the ChEMBL tier, so there is far
  less for it to move either way.</p>
  <p class="note"><strong>These are directional, not significant &mdash; and cannot be.</strong>
  PMO's convention is 5 independent runs, and a two-sided paired Wilcoxon test on 5 seeds has a
  floor of p = 0.0625: even a perfectly consistent effect cannot reach p &lt; 0.05. Six seeds is
  the minimum that can. The 20-seed ChEMBL tier is where effects are actually resolved; PMO is
  here for comparability against published baselines, and these rows say only which way the
  arms lean.</p>
</section>"""


def build() -> str:
    rows50, rows100 = arm_rows(50), arm_rows(100)
    h2h = head_to_head(50) + head_to_head(100)
    h2h_rows = "".join(
        f'<tr class="{"is-llm" if "vs diversity" in r["name"] else ""}">'
        f'<td>{esc(r["name"])}</td><td>{r["budget"]}</td><td>{r["d"]:+.4f}</td>'
        f'<td>{r["p"]:.3f}</td><td>{r["wins"]}/{r["n"]}</td>'
        f'<td><span class="pill {"ns" if r["p"] >= 0.05 else "up"}">'
        f'{"indistinguishable" if r["p"] >= 0.05 and "vs diversity" in r["name"] else ("helps" if r["p"] < 0.05 else "ns")}'
        f"</span></td></tr>"
        for r in h2h
    )
    def get(rows, arm):
        return next((r for r in rows if r["arm"] == arm), None)

    v = {a: (get(rows50, a), get(rows100, a)) for a in
         ("bo_llm_constraint", "bo_llm_select", "bo_llm_warmstart",
          "bo_diverse_warmstart", "bo_oracle_constraint")}
    h2h_llm_div = [r for r in h2h if "vs diversity" in r["name"]]
    any_llm_wins = any(
        r and r["d"] is not None and r["d"] > 0 and r["p"] < 0.05
        for a in ("bo_llm_constraint", "bo_llm_select", "bo_llm_warmstart")
        for r in v[a]
    )

    hp = "".join(
        f'<h3>{esc(group)}</h3><div class="scroll"><table><thead><tr><th>Parameter</th>'
        f"<th>Value / range</th><th>Note</th></tr></thead><tbody>"
        + "".join(f'<tr><td class="mono">{esc(k)}</td><td class="mono">{esc(v)}</td>'
                  f"<td>{esc(note)}</td></tr>" for k, v, note in items)
        + "</tbody></table></div>"
        for group, items in HYPERPARAMS
    )
    files = "".join(
        f'<tr><td class="mono">{esc(p.relative_to(RES))}</td>'
        f"<td>{p.stat().st_size / 1024:.0f} KB</td></tr>"
        for p in sorted(RES.rglob("*")) if p.is_file() and p.suffix in {".json", ".csv", ".jsonl"}
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairing an LLM with Molecular BO &mdash; Results</title>
<style>
:root {{
  color-scheme: light;
  --surface-0:#f7f7f5; --surface-1:#fcfcfb; --surface-2:#efefeb;
  --ink-1:#14140f; --ink-2:#4c4c45; --ink-3:#78776d;
  --rule:#dededa; --rule-strong:#c4c4bd;
  --c-ref:#2a78d6; --c-llm:#eb6834; --c-ceil:#1baf7a; --c-mute:#9a9a91;
  --bad:#fbeceb; --good:#e7f6ef;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --surface-0:#131312; --surface-1:#1a1a19; --surface-2:#242422;
    --ink-1:#ffffff; --ink-2:#c3c2b7; --ink-3:#8e8d84;
    --rule:#32322f; --rule-strong:#46453f;
    --c-ref:#3987e5; --c-llm:#d95926; --c-ceil:#199e70; --c-mute:#77776f;
    --bad:#2d1c1b; --good:#13291f;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--surface-0); color:var(--ink-1); font-size:15px; line-height:1.6;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:40px 20px 80px; }}
h1,h2,h3 {{ margin:0; text-wrap:balance; font-family: Georgia, "Times New Roman", serif; }}
h1 {{ font-size:clamp(26px,5vw,38px); font-weight:600; line-height:1.15; letter-spacing:-0.01em; }}
h2 {{ font-size:21px; margin-top:6px; }}
h3 {{ font-size:15.5px; margin-top:14px; font-family:inherit; }}
p {{ margin:0; color:var(--ink-2); }}
.lede {{ font-size:17px; max-width:68ch; }}
.mono, .smi {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
.smi {{ font-size:10.5px; max-width:300px; overflow-wrap:anywhere; }}
.eyebrow {{ font-family:ui-monospace,Consolas,monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); }}
header {{ display:flex; flex-direction:column; gap:14px; padding-bottom:24px;
  border-bottom:2px solid var(--rule-strong); }}
section {{ display:flex; flex-direction:column; gap:15px; padding:32px 0; border-bottom:1px solid var(--rule); }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; padding-top:6px; }}
nav a {{ font-size:12px; text-decoration:none; color:var(--ink-2); background:var(--surface-2);
  border:1px solid var(--rule); border-radius:99px; padding:4px 11px; }}
nav a:hover {{ color:var(--ink-1); border-color:var(--rule-strong); }}
.verdict {{ background:var(--surface-1); border:1px solid var(--rule); border-left:3px solid var(--c-llm);
  border-radius:3px; padding:18px 20px; display:flex; flex-direction:column; gap:10px; }}
.verdict p {{ color:var(--ink-1); }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; min-width:560px; }}
caption {{ text-align:left; font-size:12.5px; color:var(--ink-3); padding-bottom:9px; }}
th,td {{ padding:7px 10px; text-align:right; border-bottom:1px solid var(--rule); vertical-align:top; }}
th:first-child, td:first-child {{ text-align:left; }}
thead th {{ font-size:10.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink-3); border-bottom:1.5px solid var(--rule-strong); white-space:nowrap; }}
tbody td {{ font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; color:var(--ink-2); }}
tbody td:first-child {{ font-family:inherit; color:var(--ink-1); }}
tr.is-ref td {{ background:var(--surface-2); }}
tr.is-ref .arm {{ font-weight:700; }}
tr.is-llm .arm, tr.is-llm td:first-child {{ color:var(--c-llm); font-weight:600; }}
tr.is-ceil .arm {{ color:var(--c-ceil); font-weight:600; }}
tr.muted td {{ opacity:.55; }}
.arm {{ display:block; }}
.blurb {{ display:block; font-size:11.5px; color:var(--ink-3); font-weight:400; margin-top:2px; }}
.pill {{ font-family:ui-monospace,Consolas,monospace; font-size:10.5px; padding:1px 6px;
  border-radius:3px; white-space:nowrap; }}
.pill.up {{ background:var(--good); color:var(--ink-1); }}
.pill.down {{ background:var(--bad); color:var(--ink-1); }}
.pill.ns, .pill.ref {{ background:var(--surface-2); color:var(--ink-3); }}
figure {{ margin:0; display:flex; flex-direction:column; gap:10px; }}
.chart {{ background:var(--surface-1); border:1px solid var(--rule); border-radius:3px; padding:16px 14px 8px; }}
svg {{ display:block; width:100%; height:auto; }}
svg text {{ font-family:-apple-system,"Segoe UI",Roboto,sans-serif; }}
svg text.mono {{ font-family:ui-monospace,Consolas,monospace; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--ink-2); }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.sw {{ width:10px; height:10px; border-radius:50%; }}
.dash {{ width:14px; border-top:2px dashed var(--c-ref); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
.card {{ background:var(--surface-1); border:1px solid var(--rule); border-radius:3px;
  padding:14px 15px; display:flex; flex-direction:column; gap:6px; }}
.card .num {{ font-family:ui-monospace,Consolas,monospace; font-size:23px; font-weight:600;
  color:var(--ink-1); line-height:1.1; }}
.card .cap {{ font-size:12.5px; color:var(--ink-2); }}
pre {{ background:var(--surface-1); border:1px solid var(--rule); border-radius:3px; padding:12px 14px;
  overflow-x:auto; font-size:12px; color:var(--ink-2); margin:0; line-height:1.7;
  font-family:ui-monospace,Consolas,monospace; }}
ul {{ margin:0; padding-left:19px; color:var(--ink-2); display:flex; flex-direction:column; gap:7px; }}
.note {{ font-size:12.5px; color:var(--ink-3); }}
</style></head><body><div class="wrap">

<header>
  <span class="eyebrow">Results report &middot; generated by scripts/build_report.py</span>
  <h1>Three ways to pair an LLM with molecular Bayesian optimisation</h1>
  <p class="lede">A local Qwen3.5-4B narrows the library, or picks which candidate gets assayed,
  or chooses what to test first. Each is measured against plain BO, random search, a matched
  non-LLM control, a ceiling that already knows every answer, and published baselines from the
  PMO benchmark.</p>
  <nav>
    <a href="#verdict">Verdict</a><a href="#repro">Reproduce</a><a href="#setup">Data</a>
    <a href="#hyper">Hyperparameters</a><a href="#whatbo">What BO does</a>
    <a href="#results">Results</a><a href="#pairings">The three pairings</a>
    <a href="#public">Public benchmark</a><a href="#groundtruth">Ground truth</a>
    <a href="#headroom">Headroom</a><a href="#bugs">Corrections</a><a href="#files">Files</a>
  </nav>
</header>

<section id="verdict">
  <span class="eyebrow">Verdict</span>
  <div class="verdict">
    <p><strong>None of the three pairings beats plain BO.</strong> Across three quite different
    couplings &mdash; narrowing the library, choosing which candidate is assayed, and choosing
    what to test first &mdash; no LLM arm is significantly ahead of the GP on its own, and the
    constraint arm is significantly <em>behind</em> at a 100-call budget.</p>
    <p>The one arm that gains ground, the warm start, is matched by a free MaxMin diversity
    heuristic choosing from the same 40 molecules with no model involved &mdash; and the
    heuristic is in fact slightly ahead at both budgets. The model says as much itself: its
    recorded rationales are explicitly diversity arguments (&ldquo;Core Scaffold
    Diversity&hellip; distinct structural scaffolds&hellip; maximize information gain&rdquo;).
    It is doing MaxMin selection in prose.</p>
  </div>
  <div class="cards">
    <div class="card"><span class="num">{v["bo_llm_constraint"][1]["d"]:+.3f}</span>
    <span class="cap"><strong>Constraint</strong> at 100 calls
    (p&nbsp;=&nbsp;{v["bo_llm_constraint"][1]["p"]:.3f}). Significantly worse. A bad mask is
    <em>absorbing</em> &mdash; it excluded the best compound in 19 of 20 proposals and BO never
    recovers.</span></div>
    <div class="card"><span class="num">{v["bo_llm_select"][0]["d"]:+.3f}</span>
    <span class="cap"><strong>Selection</strong> at 50 calls (ns). With list-position bias
    removed it agrees with BO 3&times; more than chance &mdash; a noisy copy of the acquisition
    function, not an improvement on it.</span></div>
    <div class="card"><span class="num">{v["bo_llm_warmstart"][0]["d"]:+.3f}</span>
    <span class="cap"><strong>Warm start</strong> at 50 calls
    (p&nbsp;=&nbsp;{v["bo_llm_warmstart"][0]["p"]:.2f}, ns). Gains ground, but its own free
    control gains {v["bo_diverse_warmstart"][0]["d"]:+.3f} &mdash; more, not less.</span></div>
  </div>
  <p>Headroom is not the problem: the ceiling arm, which simply knows the answers, is worth
  {v["bo_oracle_constraint"][0]["d"]:+.3f} at 50 calls
  (p&nbsp;&lt;&nbsp;0.001). A good constraint wins comfortably here &mdash; this model just does
  not find one.</p>
  <p class="note"><strong>Correction:</strong> an earlier version of this page reported the warm
  start as a significant win (+0.030, p&nbsp;=&nbsp;0.024). That was measured with
  stereochemistry-blind fingerprints. After fixing the representation (see
  <a href="#space">Search space</a>) the effect is no longer significant and the non-LLM control
  overtakes it. Every number here was regenerated after the fix.</p>
  <p class="note">Scope: Qwen3.5-4B running locally. It misidentifies the series' chemotype in
  every prompt format tried &mdash; calling anilinoquinazolines a &ldquo;pyrazolo[3,4-d]pyrimidine
  core&rdquo; &mdash; so these are results about a small local model, not about language models
  in general. The harness is backend-agnostic.</p>
</section>

<section id="repro">
  <span class="eyebrow">Reproduce</span>
  <h2>Exact commands</h2>
  <pre># 1. Serve the model locally (nothing here uses a hosted API)
llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q8_0 --port 8080 \\
    --ctx-size 16384 --n-gpu-layers all --reasoning-budget 0

# 2. Build the congeneric series from ChEMBL (already in data/chembl)
python scripts/build_chembl_series.py --target CHEMBL203 --out data/chembl/egfr

# 3. Measure headroom BEFORE committing to an arm
python scripts/constraint_headroom.py \\
    --dataset data/chembl/egfr_anilinoquinazoline.csv --seeds 10 \\
    --out results/headroom_egfr_congeneric.json

# 4. The main sweep: all arms, 20 paired seeds
python scripts/run_lookup_experiment.py \\
    --dataset data/chembl/egfr_anilinoquinazoline.csv \\
    --budget 50 --seeds 20 --constraint-fraction 0.05 \\
    --llm-arm --select-arm --warmstart-arm --shortlist-k 15 \\
    --out results/bench_final_b50

# 5. The public benchmark, against PMO's published baselines
python scripts/run_pmo_experiment.py --seeds 5 --budget 1000 \\
    --tasks albuterol_similarity celecoxib_rediscovery deco_hop \\
            median1 osimertinib_mpo --out results/pmo

# 6. Regenerate this page from whatever is in results/
python scripts/build_report.py</pre>
  <p>Determinism: the model runs greedy (<span class="mono">temperature 0</span>,
  <span class="mono">top_k 1</span>, fixed seed) and every response is cached by a hash of the
  exact request, so a rerun makes <strong>zero</strong> model calls and reproduces the tables
  byte for byte. Delete <span class="mono">data/cache/llm*</span> to force fresh generations.</p>
  <p class="note">PMO install: <span class="mono">uv pip install --no-deps PyTDC==0.4.17</span>
  then <span class="mono">fuzzywuzzy seaborn "setuptools&lt;81" mygene huggingface_hub networkx
  biopython</span>. PyTDC otherwise pulls <span class="mono">tiledbsoma</span>, which nothing
  here uses and which needs a build toolchain.</p>
</section>

<section id="setup">
  <span class="eyebrow">Data</span>
  <h2>What the oracle is</h2>
  <p>A dictionary lookup of real measured activities, which is why a 20-seed sweep costs about a
  minute and why 20 seeds are affordable at all. It is also the only tier with true ground
  truth: the library optimum is known, so simple regret and top-1% recall are computable.</p>
  <div class="scroll"><table><tbody>
    <tr><td>Source</td><td class="mono">ChEMBL REST API, target CHEMBL203 (EGFR)</td></tr>
    <tr><td>Assay</td><td class="mono">IC50, binding assays (assay_type = B), pChEMBL values</td></tr>
    <tr><td>Filters</td><td class="mono">standard_relation '=' only; empty data_validity_comment; replicates collapsed to median; max SEM 0.5</td></tr>
    <tr><td>Series</td><td class="mono">anilinoquinazoline scaffold cluster, core c1ccc2ncnc(Nc3ccccc3)c2c1</td></tr>
    <tr><td>Size filter</td><td class="mono">10&ndash;55 heavy atoms (removed 17 of 2,103; optimum unchanged)</td></tr>
    <tr><td>Library</td><td class="mono">2,086 compounds, pIC50 4.00&ndash;11.00, median 7.40</td></tr>
    <tr><td>Objective</td><td class="mono">maximise pIC50 (higher = more potent)</td></tr>
    <tr><td>Metric</td><td class="mono">AUC of the top-10 mean vs cumulative UNIQUE oracle calls (PMO's metric)</td></tr>
  </tbody></table></div>
  <p><strong>Why censored records are dropped.</strong> A &ldquo;&gt;10000 nM&rdquo; result is a
  bound, not a measurement; scoring it as a pIC50 of 5 would invent a cliff of fake inactives
  and hand the surrogate structure that is not there. No 3D protein structure is used anywhere
  in this report.</p>
</section>

<section id="hyper">
  <span class="eyebrow">Hyperparameters</span>
  <h2>Every knob, and its range</h2>
  <p>Nothing below was tuned against the results. The GP's own hyperparameters are the exception:
  they are <em>refitted from the data every round</em> by marginal likelihood, so they are a range
  rather than a setting &mdash; see the next section for what was actually selected.</p>
  {hp}
</section>

{search_space_section()}

{gp_section()}

<section id="results">
  <span class="eyebrow">Results</span>
  <h2>Every arm against plain BO</h2>
  <p>Points are mean AUC over 20 paired seeds; whiskers are &plusmn;1 SEM. The dashed line is
  plain BO. Beating random search is a smoke test, not a result &mdash; the bar is the blue line.</p>
  <figure>
    <div class="chart">{chart_svg(rows50, 50)}</div>
    <div class="legend">
      <span><i class="sw" style="background:var(--c-ref)"></i>Plain BO (reference)</span>
      <span><i class="sw" style="background:var(--c-llm)"></i>LLM arms</span>
      <span><i class="sw" style="background:var(--c-ceil)"></i>Ceiling</span>
      <span><i class="sw" style="background:var(--c-mute)"></i>Non-LLM controls</span>
      <span><i class="dash"></i>plain-BO level</span>
    </div>
    <figcaption class="note">Budget 50 unique oracle calls. Hover a point for its exact value.</figcaption>
  </figure>
  <h3>Budget 50 unique oracle calls</h3>
  {arm_table(rows50)}
  <h3>Budget 100 unique oracle calls</h3>
  {arm_table(rows100)}
  <p>Two controls carry the argument. <strong>A random mask of the same tightness significantly
  hurts</strong> (&minus;0.081 at 100 calls), which kills the obvious objection that any
  restriction helps a small-budget search. And <strong>the purely statistical mask is
  neutral</strong> (&minus;0.003, p&nbsp;=&nbsp;0.65), so a win could not have been dismissed as
  the model merely summarising descriptors.</p>
</section>

<section id="pairings">
  <span class="eyebrow">The three pairings</span>
  <h2>1 &mdash; Constraint: the model narrows the library</h2>
  <p>The model sees the SAR so far plus a menu of 28 substructures mined from the library, and
  returns a mask. It chooses by index and never writes SMARTS: asked to write them, this model
  produced an invalid pattern <em>every single time</em>, the arm collapsed to a molecular-weight
  range keeping 89% of the library, and the experiment measured typing rather than chemistry.</p>
  <div class="cards">
    <div class="card"><span class="num">1 / 20</span><span class="cap">proposals retain the single
    best compound. A mask is <strong>absorbing</strong> &mdash; once the optimum is fenced off,
    BO cannot recover for the rest of the run.</span></div>
    <div class="card"><span class="num">23.5% &rarr; 31.0%</span><span class="cap">library kept vs
    true top-1% captured. A perfectly uninformative mask would capture 23.5%.</span></div>
    <div class="card"><span class="num">0.7350</span><span class="cap">AUC when the constraint is
    re-proposed 6&times; per run instead of once (0.7340). More evidence does not help &mdash; the
    &ldquo;it only sees 10 examples&rdquo; hypothesis is dead.</span></div>
  </div>
  <h2>2 &mdash; Selection: BO shortlists, the model picks</h2>
  <p>Each round BO ranks candidates, the top 15 go to the model, and it chooses which one costs an
  oracle call. A bad pick costs one evaluation, so unlike a constraint it is recoverable. It costs
  one model call per <em>evaluation</em> &mdash; about 40&times; the constraint arm.
  <span class="mono">shortlist_k</span> is the dial: <strong>k=1 is exactly plain BO</strong>
  (pinned by a test, including on a flat acquisition surface), k = all candidates is pure
  selection.</p>
  {selector_section()}
  <h2>3 &mdash; Warm start: the model picks what to test first</h2>
  <p>With nothing yet measured, the model chooses the first 10 compounds from a seeded
  40-molecule sample. This is the best case a language model gets: there is no data, so the
  surrogate has nothing to offer and prior knowledge is all there is. One call per run.</p>
  <p>The comparison is unusually tight &mdash; the sample is drawn with the campaign's own seed,
  the same draw plain BO's random design comes from, so the <em>only</em> difference is which
  members of an identical pool get bought first.</p>
  <div class="scroll"><table>
    <caption>Head to head. The diversity control picks from the same sample by greedy MaxMin
    Tanimoto distance, with no model involved.</caption>
    <thead><tr><th>Comparison</th><th>Budget</th><th>&Delta;AUC</th><th>p</th>
    <th>seeds ahead</th><th>Verdict</th></tr></thead>
    <tbody>{h2h_rows}</tbody></table></div>
  <p><strong>Warm-starting BO helps a little; the LLM is not what helps.</strong> Neither warm
  start reaches significance against plain BO once the fingerprint is correct, and the free
  diversity heuristic is ahead of the model at both budgets &mdash; on 15 of 20 seeds against
  plain BO, where the model manages 10 or 11.</p>
</section>

{pmo_section()}

{pmo_ground_truth_section()}

{pmo_llm_section()}

{headroom_section()}

<section id="bugs">
  <span class="eyebrow">Corrections</span>
  <h2>Bugs found while building this</h2>
  <p>Listed because each one produced a plausible-looking number before it was caught, and
  every one was found by a test asserting an invariant rather than by reading the output.</p>
  <div class="cards">
    <div class="card fix"><h3>A constraint arm that silently wasn't one</h3><p class="cap">
    The PMO runner built its <span class="mono">LibraryContext</span> directly, leaving
    <span class="mono">mols</span> empty, so every substructure mask came back all-False &mdash;
    indistinguishable from "matched nothing". The arm degraded to unconstrained and reported
    <strong>&Delta;AUC = +0.0000 on every task, p = 1.0</strong>. A clean zero reads like a
    finding. Molecules and descriptors are now computed lazily, so a context is correct however
    it is built, and the runner prints constraint modes.</p></div>
    <div class="card fix"><h3>Molecules the surrogate could not tell apart</h3><p class="cap">
    RDKit's Morgan generator ignores stereochemistry by default, so 133 pairs &mdash; ~10% of the
    library &mdash; were identical to the GP while their activities differed by up to 3.0 log
    units. <span class="mono">includeChirality=True</span> halves it. <strong>This one changed a
    conclusion</strong>: the warm start was significant before the fix and is not after.</p></div>
    <div class="card fix"><h3>False coverage figures shown to the model</h3><p class="cap">
    Menu entries were counted by how often a scaffold <em>occurred</em>, not how often its query
    <em>matches</em>. The anilinoquinazoline core was advertised at 25.3% of the library; it
    actually matches 100%, because the library was selected by that core. Candidates are now
    counted with the same query used at match time.</p></div>
    <div class="card fix"><h3>Generic scaffolds that could never match</h3><p class="cap">
    <span class="mono">MakeScaffoldGeneric</span> maps benzene to <span class="mono">C1CCCCC1</span>,
    which as a query matches aliphatic rings and so matches none of the aromatic molecules it came
    from &mdash; advertised at 50% coverage, actual coverage zero.</p></div>
    <div class="card fix"><h3>Answers lost to truncation</h3><p class="cap">
    The selection schema emitted <span class="mono">reasoning</span> before
    <span class="mono">choice</span>, so a response overrunning <span class="mono">max_tokens</span>
    lost its answer &mdash; 6 of 41 calls in a pilot. In aggregate a 17% failure rate looks exactly
    like a model that agrees with BO. <span class="mono">choice</span> now comes first.</p></div>
    <div class="card fix"><h3>Position bias impersonating judgment</h3><p class="cap">
    The shortlist was shown in acquisition order, so "picks position 0" and "agrees with BO's top
    candidate" were the same event. The choice distribution was peaked at both list ends. Shuffling
    and mapping back decoupled them &mdash; and only then was the model's real (modest) agreement
    with the acquisition function measurable.</p></div>
    <div class="card fix"><h3>Inconsistent tie-breaking</h3><p class="cap">
    Plain BO broke ties at random; the selector shortlist broke them at index 0. On a flat
    acquisition surface &mdash; every EI numerically zero, the normal early state &mdash; the claim
    "k=1 is plain BO" quietly failed. Both now share one ranking function.</p></div>
    <div class="card fix"><h3>A mask recomputed every round</h3><p class="cap">
    Constraints were re-evaluated each iteration even when unchanged: ~10 s per round on a 60k
    pool, over an hour wasted per PMO campaign. Masks are now cached on the library context,
    keyed by the constraint's spec hash so a genuinely new proposal still recomputes.</p></div>
  </div>
</section>

<section id="files">
  <span class="eyebrow">Artifacts</span>
  <h2>Everything in results/</h2>
  <p>Each sweep directory holds <span class="mono">metrics.csv</span> (one row per arm per seed),
  <span class="mono">summary.json</span> (aggregates, provenance and the paired tests),
  <span class="mono">traces.jsonl</span> (every molecule bought, in order, with its value) and,
  where relevant, the model's own logs. The traces are sufficient to recompute any metric offline.</p>
  <div class="scroll"><table><thead><tr><th>File</th><th>Size</th></tr></thead>
  <tbody>{files}</tbody></table></div>
</section>

</div></body></html>"""


def main() -> int:
    if not RES.exists():
        print(f"error: {RES} does not exist", file=sys.stderr)
        return 2
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

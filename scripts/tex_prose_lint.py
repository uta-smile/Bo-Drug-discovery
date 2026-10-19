#!/usr/bin/env python3
"""LaTeX-aware prose linter.

Reports stylistic tells in manuscript prose while respecting LaTeX structure:
math, citations, labels, verbatim blocks and tables are masked before any
check runs, so no finding ever points at markup.

Report-only. This script never modifies the input file.

Usage:
    python scripts/tex_prose_lint.py report/paper.tex
    python scripts/tex_prose_lint.py report/paper.tex --json
    python scripts/tex_prose_lint.py report/paper.tex --only uniformity,hedges
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Masking: blank out everything that is not prose, preserving character
# offsets so that every finding maps back to a real line number.
# --------------------------------------------------------------------------

VERBATIM_ENVS = ("verbatim", "lstlisting", "minted", "Verbatim", "alltt")
MATH_ENVS = (
    "equation", "align", "gather", "multline", "eqnarray",
    "displaymath", "array", "split", "alignat", "flalign",
)
TABLE_ENVS = ("tabular", "tabularx", "tabulary", "longtable", "supertabular")

# Commands whose arguments are identifiers, not prose.
OPAQUE_CMDS = (
    r"cite[a-zA-Z]*", r"[Cc]ref", r"ref", r"eqref", r"pageref", r"autoref",
    r"label", r"bibliography[a-zA-Z]*", r"includegraphics", r"input",
    r"include", r"url", r"href", r"texttt", r"path", r"newcommand",
    r"renewcommand", r"def", r"usepackage", r"documentclass", r"setlength",
    r"addtolength", r"DeclareMathOperator", r"newtheorem", r"bibitem",
    r"begin", r"end",
)


def _blank(text: str, start: int, end: int) -> str:
    """Replace [start, end) with spaces, keeping newlines so lines still align."""
    chunk = "".join("\n" if c == "\n" else " " for c in text[start:end])
    return text[:start] + chunk + text[end:]


def _mask_pattern(text: str, pattern: re.Pattern) -> str:
    out = text
    for m in pattern.finditer(text):
        out = _blank(out, m.start(), m.end())
    return out


def _mask_balanced_arg(text: str, cmd_pattern: str) -> str:
    """Mask \\cmd{...} including nested braces, leaving the rest intact."""
    out = text
    pattern = r"\\(?:" + cmd_pattern + r")\s*\*?\s*(?:\[[^\]]*\])*\s*\{"
    for m in re.finditer(pattern, text):
        depth = 0
        i = m.end() - 1
        while i < len(text):
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and text[i - 1] != "\\":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out = _blank(out, m.start(), min(i + 1, len(text)))
    return out


def mask_non_prose(text: str) -> str:
    """Blank every non-prose region, preserving offsets."""
    t = text

    # Preamble: everything before \begin{document}.
    begin = re.search(r"\\begin\{document\}", t)
    if begin:
        t = _blank(t, 0, begin.end())

    # Comments (unescaped %).
    t = _mask_pattern(t, re.compile(r"(?<!\\)%[^\n]*"))

    # Verbatim-like, math, and table environments.
    for env in VERBATIM_ENVS + MATH_ENVS + TABLE_ENVS:
        t = _mask_pattern(t, re.compile(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
            re.DOTALL))

    # Display and inline math delimiters.
    t = _mask_pattern(t, re.compile(r"\\\[.*?\\\]", re.DOTALL))
    t = _mask_pattern(t, re.compile(r"\\\(.*?\\\)", re.DOTALL))
    t = _mask_pattern(t, re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$", re.DOTALL))
    t = _mask_pattern(t, re.compile(r"(?<!\\)\$(?:[^$\\]|\\.)*?(?<!\\)\$",
                                    re.DOTALL))

    # Opaque command arguments.
    t = _mask_balanced_arg(t, "|".join(OPAQUE_CMDS))

    # Bare control sequences left over (\section, \item, \textbf ...): drop the
    # command token but keep any brace-delimited prose argument.
    t = _mask_pattern(t, re.compile(r"\\[a-zA-Z@]+\*?"))
    t = _mask_pattern(t, re.compile(r"\\[^a-zA-Z]"))

    return t


# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

ABBREV = {
    "e.g", "i.e", "cf", "vs", "etc", "al", "fig", "eq", "sec", "tab",
    "approx", "resp", "ref", "no", "dr", "prof", "st", "ca", "viz",
}

_SENT_END = re.compile(r"[.!?](?=[\s\)\"']|$)")


@dataclass
class Sentence:
    text: str
    start: int
    line: int
    para: int

    @property
    def words(self) -> list:
        return re.findall(r"[A-Za-z][A-Za-z'\-]*", self.text)

    @property
    def n_words(self) -> int:
        return len(self.words)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def segment(masked: str, original: str) -> list:
    """Split masked prose into sentences, tracking line numbers."""
    sentences = []
    para_idx = 0
    pos = 0
    start = 0
    for m in _SENT_END.finditer(masked):
        head = masked[start:m.end()]
        # Skip abbreviations and decimals.
        tail = re.search(r"([A-Za-z\.]+)\.$", head.strip())
        if tail and tail.group(1).lower().strip(".") in ABBREV:
            continue
        if m.start() > 0 and masked[m.start() - 1].isdigit():
            nxt = masked[m.end():m.end() + 1]
            if nxt.isdigit():
                continue
        raw = head.strip()
        if raw:
            off = start + (len(head) - len(head.lstrip()))
            para_idx += masked[pos:off].count("\n\n")
            pos = off
            sentences.append(Sentence(
                text=re.sub(r"\s+", " ", raw),
                start=off,
                line=line_of(original, off),
                para=para_idx,
            ))
        start = m.end()
    return [s for s in sentences if s.n_words >= 3]


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------

@dataclass
class Finding:
    check: str
    severity: str          # high | medium | low
    line: int
    excerpt: str
    message: str


def _excerpt(s: str, limit: int = 110) -> str:
    s = s.strip()
    return s if len(s) <= limit else s[:limit - 1] + "..."


# --- check: sentence-length uniformity ------------------------------------

def check_uniformity(sents: list, window: int = 4,
                     stdev_floor: float = 3.0) -> list:
    """Flag metronomic runs.

    Windows never cross a paragraph boundary: a run spanning a caption, a
    section break or two unrelated paragraphs is an artifact of concatenation,
    not a cadence the reader experiences as uniform.
    """
    out = []
    by_para = {}
    for s in sents:
        by_para.setdefault(s.para, []).append(s)

    for para in sorted(by_para):
        group = by_para[para]
        lens = [s.n_words for s in group]
        i = 0
        while i + window <= len(lens):
            chunk = lens[i:i + window]
            sd = statistics.pstdev(chunk)
            if sd < stdev_floor and min(chunk) >= 12:
                out.append(Finding(
                    "uniformity", "high", group[i].line,
                    _excerpt(group[i].text),
                    "%d consecutive sentences of %s words (sd=%.1f). Uniform "
                    "cadence reads mechanical; vary by splitting one and "
                    "merging two." % (window, chunk, sd),
                ))
                i += window
            else:
                i += 1
    return out


# --- check: repeated sentence openings ------------------------------------

CONNECTIVES = {
    "however", "moreover", "furthermore", "additionally", "importantly",
    "notably", "consequently", "therefore", "thus", "indeed", "overall",
    "crucially", "specifically", "ultimately", "nonetheless",
}


def check_openings(sents: list) -> list:
    out = []
    by_para = {}
    for s in sents:
        by_para.setdefault(s.para, []).append(s)

    for para, group in by_para.items():
        counts = {}
        for s in group:
            w = s.words[0].lower() if s.words else ""
            counts.setdefault(w, []).append(s)
        for word, hits in counts.items():
            if len(hits) >= 3 and word:
                out.append(Finding(
                    "openings", "medium", hits[0].line, _excerpt(hits[0].text),
                    "%d sentences in this paragraph open with \"%s\" (lines %s)."
                    % (len(hits), word,
                       ", ".join(str(h.line) for h in hits)),
                ))

    # Runs of connective-opened sentences anywhere.
    run = []
    for s in sents:
        first = s.words[0].lower() if s.words else ""
        if first in CONNECTIVES:
            run.append(s)
        else:
            if len(run) >= 3:
                out.append(Finding(
                    "openings", "medium", run[0].line, _excerpt(run[0].text),
                    "%d consecutive sentences open with a discourse "
                    "connective. Cut all but the one that marks a real turn."
                    % len(run),
                ))
            run = []
    if len(run) >= 3:
        out.append(Finding(
            "openings", "medium", run[0].line, _excerpt(run[0].text),
            "%d consecutive sentences open with a discourse connective."
            % len(run),
        ))
    return out


# --- check: formulaic constructions ---------------------------------------

FORMULAIC = [
    (r"\bnot only\b.{0,80}?\bbut also\b", "high",
     "\"not only ... but also\" - split into two sentences."),
    (r"\bit is (?:worth noting|important to note|notable)\b", "high",
     "Throat-clearing: state the point directly."),
    (r"\bplays? an? (?:crucial|key|vital|important|pivotal) role\b", "high",
     "Vague: say what it actually does."),
    (r"\bdelve[sd]? into\b", "high", "Replace with \"examine\" or \"analyze\"."),
    (r"\bin the realm of\b", "high", "Replace with \"in\"."),
    (r"\ba testament to\b", "high", "Replace with the evidence itself."),
    (r"\bserves? as a\b", "medium", "Usually replaceable with \"is\"."),
    (r"\bleverag(?:e|es|ed|ing)\b", "medium", "Prefer \"use\" or \"exploit\"."),
    (r"\bunderscore[sd]?\b", "medium", "Prefer \"shows\" or \"confirms\"."),
    (r"\bshed(?:s|ding)? light on\b", "medium", "Prefer \"explains\"."),
    (r"\bpaves? the way\b", "medium", "Vague forward-looking filler."),
    (r"\bcomprehensive (?:analysis|study|evaluation|overview)\b", "medium",
     "\"Comprehensive\" is unverifiable; name the scope instead."),
    (r"\bintricate\b|\bmultifaceted\b|\bnuanced\b", "medium",
     "Ornamental adjective; cut or make specific."),
    (r"\bstate-of-the-art\b", "low", "Acceptable once; flag if it recurs."),
]


def check_formulaic(sents: list) -> list:
    out = []
    for s in sents:
        for pat, sev, msg in FORMULAIC:
            m = re.search(pat, s.text, re.IGNORECASE)
            if m:
                out.append(Finding("formulaic", sev, s.line,
                                   _excerpt(s.text),
                                   "\"%s\": %s" % (m.group(0), msg)))
    return out


# --- check: hedge stacks ---------------------------------------------------

HEDGES = (r"\b(may|might|could|possibly|potentially|perhaps|appears? to|"
          r"seems? to|relatively|somewhat|fairly|arguably|largely|generally|"
          r"tends? to|suggests? that)\b")


def check_hedges(sents: list) -> list:
    out = []
    for s in sents:
        hits = re.findall(HEDGES, s.text, re.IGNORECASE)
        if len(hits) >= 2:
            out.append(Finding(
                "hedges", "high", s.line, _excerpt(s.text),
                "%d hedges in one sentence (%s). Keep at most the one that "
                "reflects genuine uncertainty."
                % (len(hits), ", ".join(h.lower() for h in hits)),
            ))
    return out


# --- check: filler and intensifiers ---------------------------------------

# "rather than" is a legitimate contrastive, so only bare "rather" counts.
FILLER = (r"\b(actually|basically|essentially|really|very|quite|simply|"
          r"clearly|obviously|a bit|rather(?! than)|extremely|highly|"
          r"incredibly)\b")


def check_filler(sents: list) -> list:
    out = []
    for s in sents:
        for m in re.finditer(FILLER, s.text, re.IGNORECASE):
            out.append(Finding(
                "filler", "medium", s.line, _excerpt(s.text),
                "Filler \"%s\": delete unless it changes meaning." % m.group(0),
            ))
    return out


# --- check: nominalizations ------------------------------------------------

NOMINAL = re.compile(
    r"\b(perform(?:ed|s)?|conduct(?:ed|s)?|carr(?:y|ied) out|provide[sd]?|"
    r"make[s]?|made|undertak(?:e|es|en))\s+(?:an?|the)\s+"
    r"([a-z]+(?:tion|ment|ance|ence|ysis|ison|ation))\b", re.IGNORECASE)


def check_nominalizations(sents: list) -> list:
    out = []
    for s in sents:
        for m in NOMINAL.finditer(s.text):
            out.append(Finding(
                "nominalization", "medium", s.line, _excerpt(s.text),
                "\"%s\": put the action in the verb (e.g. \"we analyzed\" "
                "over \"we performed an analysis\")." % m.group(0),
            ))
    return out


# --- check: passive voice --------------------------------------------------

PASSIVE = re.compile(
    r"\b(?:is|are|was|were|been|being|be)\s+(?:[a-z]+ly\s+)?"
    r"([a-z]+(?:ed|en))\b(?!\s+(?:to|that))", re.IGNORECASE)


def check_passive(sents: list) -> list:
    out = []
    for s in sents:
        hits = PASSIVE.findall(s.text)
        if len(hits) >= 2:
            out.append(Finding(
                "passive", "low", s.line, _excerpt(s.text),
                "%d passive constructions in one sentence (%s). Name the "
                "agent where it matters." % (len(hits), ", ".join(hits)),
            ))
    return out


# --- check: em-dash density -----------------------------------------------

def check_dashes(masked: str, sents: list) -> list:
    n_words = sum(s.n_words for s in sents)
    n_dash = len(re.findall(r"---|\u2014", masked))
    if n_words == 0:
        return []
    per_k = 1000.0 * n_dash / n_words
    if per_k > 4.0:
        return [Finding(
            "dashes", "medium", 0, "",
            "%d em-dashes across %d prose words (%.1f per 1000). Dense "
            "em-dash use is a recognisable tell; convert some to commas, "
            "colons or full stops." % (n_dash, n_words, per_k),
        )]
    return []


# --- check: terminology consistency ---------------------------------------

VARIANT_GROUPS = [
    ("hyperparameter", "hyper-parameter", "hyper parameter"),
    ("dataset", "data set", "data-set"),
    ("multitask", "multi-task", "multi task"),
    ("downstream", "down-stream"),
    ("held-out", "heldout", "held out"),
    ("single-task", "singletask", "single task"),
    ("out-of-distribution", "out of distribution"),
    ("fine-tune", "finetune", "fine tune"),
    ("acquisition function", "acquisition-function"),
    ("surrogate model", "surrogate-model"),
]


def check_terminology(masked: str) -> list:
    out = []
    low = masked.lower()
    for group in VARIANT_GROUPS:
        present = []
        for variant in group:
            pat = re.compile(r"\b" + re.escape(variant) + r"s?\b")
            hits = list(pat.finditer(low))
            if hits:
                present.append((variant, len(hits), hits[0].start()))
        if len(present) >= 2:
            present.sort(key=lambda x: -x[1])
            desc = ", ".join("\"%s\" x%d" % (v, n) for v, n, _ in present)
            out.append(Finding(
                "terminology", "high", line_of(masked, present[-1][2]), "",
                "Mixed spellings: %s. Pick one and apply it throughout." % desc,
            ))
    return out


# --------------------------------------------------------------------------

CHECKS = {
    "uniformity": lambda m, s: check_uniformity(s),
    "openings": lambda m, s: check_openings(s),
    "formulaic": lambda m, s: check_formulaic(s),
    "hedges": lambda m, s: check_hedges(s),
    "filler": lambda m, s: check_filler(s),
    "nominalization": lambda m, s: check_nominalizations(s),
    "passive": lambda m, s: check_passive(s),
    "dashes": lambda m, s: check_dashes(m, s),
    "terminology": lambda m, s: check_terminology(m),
}

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LaTeX-aware prose linter (report-only).")
    ap.add_argument("texfile", type=Path)
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output")
    ap.add_argument("--only", help="comma-separated check names")
    ap.add_argument("--dump-prose", action="store_true",
                    help="print the masked prose and exit (masking sanity check)")
    args = ap.parse_args()

    original = args.texfile.read_text(encoding="utf-8", errors="replace")
    masked = mask_non_prose(original)

    if args.dump_prose:
        sys.stdout.write(masked)
        return 0

    sents = segment(masked, original)

    selected = set(args.only.split(",")) if args.only else set(CHECKS)
    unknown = selected - set(CHECKS)
    if unknown:
        ap.error("unknown check(s): %s" % ", ".join(sorted(unknown)))

    findings = []
    for name in CHECKS:
        if name in selected:
            findings.extend(CHECKS[name](masked, sents))

    findings.sort(key=lambda f: (SEV_ORDER[f.severity], f.line))

    lens = [s.n_words for s in sents]
    stats = {
        "sentences": len(sents),
        "prose_words": sum(lens),
        "mean_sentence_words": round(statistics.mean(lens), 1) if lens else 0,
        "stdev_sentence_words": round(statistics.pstdev(lens), 1) if lens else 0,
        "min_sentence_words": min(lens) if lens else 0,
        "max_sentence_words": max(lens) if lens else 0,
    }

    if args.json:
        print(json.dumps({"file": str(args.texfile), "stats": stats,
                          "findings": [asdict(f) for f in findings]}, indent=2))
        return 0

    print("\n%s: %d prose sentences, %d words"
          % (args.texfile, stats["sentences"], stats["prose_words"]))
    print("Sentence length: mean %s, sd %s, range %s-%s"
          % (stats["mean_sentence_words"], stats["stdev_sentence_words"],
             stats["min_sentence_words"], stats["max_sentence_words"]))
    print("%d findings\n" % len(findings))

    by_check = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)

    for check, group in sorted(by_check.items(),
                               key=lambda kv: SEV_ORDER[kv[1][0].severity]):
        print("== %s (%d) %s" % (check, len(group), "=" * (50 - len(check))))
        for f in group:
            loc = ("L%d" % f.line) if f.line else "file"
            print("  [%-6s] %6s  %s" % (f.severity, loc, f.message))
            if f.excerpt:
                print("                   > %s" % f.excerpt)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

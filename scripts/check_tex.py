"""Structural checks on a LaTeX source, for a machine with no TeX installed.

    python scripts/check_tex.py report/paper.tex

This is not a compiler and does not pretend to be one. It catches the four
things that actually break a build here and that a reader cannot see by eye:

1. every ``\\cite`` key resolves to an entry in the ``.bib``,
2. every ``\\ref`` resolves to a ``\\label``,
3. every ``\\includegraphics`` file exists next to the source,
4. ``\\begin``/``\\end`` are balanced and correctly nested,
5. every row inside a ``tabular`` ends in ``\\\\`` and not a bare backslash.

Check 5 earns its place: a shell heredoc silently ate one backslash from every
row terminator in two sections of this paper, turning each row break into a line
continuation. The tables would have compiled -- into one long row. Nothing else
here would have noticed.

It also lists unused bib entries and unreferenced labels, which are not errors
but are usually a sign that a section was moved or dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|autoref|eqref)\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
BIBKEY = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
BIBFILE = re.compile(r"\\bibliography\{([^}]*)\}")
ENVS = re.compile(r"\\(begin|end)\{([^}]*)\}")
COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tex", nargs="+")
    p.add_argument("--quiet-unused", action="store_true")
    args = p.parse_args(argv)

    bad = 0
    for path in args.tex:
        src_path = Path(path)
        raw = src_path.read_text(encoding="utf-8")
        # Strip comments before anything else: a commented-out \cite is not a
        # citation, and this file deliberately carries commented placeholders.
        src = COMMENT.sub("", raw)
        here = src_path.parent
        print(f"== {path}  ({len(raw.splitlines())} lines)")

        keys: set[str] = set()
        for name in (n.strip() for m in BIBFILE.findall(src)
                     for n in m.split(",")):
            bib = here / (name if name.endswith(".bib") else name + ".bib")
            if not bib.exists():
                print(f"   MISSING bib file: {bib}")
                bad += 1
                continue
            keys |= set(BIBKEY.findall(bib.read_text(encoding="utf-8")))
        cited = {k.strip() for m in CITE.findall(src) for k in m.split(",")
                 if k.strip()}
        missing = sorted(cited - keys)
        if missing:
            print(f"   UNDEFINED citations ({len(missing)}): "
                  + ", ".join(missing))
            bad += len(missing)

        labels = set(LABEL.findall(src))
        refs = {r.strip() for m in REF.findall(src) for r in m.split(",")}
        dangling = sorted(refs - labels)
        if dangling:
            print(f"   UNDEFINED refs ({len(dangling)}): " + ", ".join(dangling))
            bad += len(dangling)

        for g in GRAPHIC.findall(src):
            cands = [here / g] + [here / (g + e) for e in (".pdf", ".png")]
            if not any(c.exists() for c in cands):
                print(f"   MISSING graphic: {g}")
                bad += 1

        stack: list[tuple[str, int]] = []
        line_of = {}
        pos = 0
        for i, line in enumerate(src.splitlines(keepends=True), 1):
            line_of[pos] = i
            pos += len(line)
        for m in ENVS.finditer(src):
            kind, env = m.group(1), m.group(2)
            ln = src.count("\n", 0, m.start()) + 1
            if kind == "begin":
                stack.append((env, ln))
            elif not stack:
                print(f"   line {ln}: \\end{{{env}}} with nothing open")
                bad += 1
            elif stack[-1][0] != env:
                print(f"   line {ln}: \\end{{{env}}} closes "
                      f"\\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
                bad += 1
                stack.pop()
            else:
                stack.pop()
        for env, ln in stack:
            print(f"   line {ln}: \\begin{{{env}}} never closed")
            bad += 1

        # Rows inside a tabular must end in a doubled backslash. A single one
        # is a line continuation, which compiles and is wrong.
        bs = chr(92)
        depth = 0
        for i, line in enumerate(src.splitlines(), 1):
            if bs + "begin{tabular}" in line:
                depth += 1
            t = line.rstrip()
            if depth > 0 and t.endswith(bs) and not t.endswith(bs * 2):
                print(f"   line {i}: tabular row ends in a single backslash: "
                      f"{t[-60:]}")
                bad += 1
            if bs + "end{tabular}" in line:
                depth = max(0, depth - 1)

        if not args.quiet_unused:
            unused = sorted(keys - cited)
            if unused:
                print(f"   (unused bib entries: {len(unused)}) "
                      + ", ".join(unused))
            orphan = sorted(labels - refs)
            if orphan:
                print(f"   (labels never referenced: {len(orphan)}) "
                      + ", ".join(orphan))
        print(f"   citations {len(cited)}/{len(keys)} bib entries, "
              f"{len(labels)} labels, {len(refs)} refs")

    print("\nOK" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

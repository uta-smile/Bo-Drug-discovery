#!/usr/bin/env bash
# Compile both submission PDFs; verify the explicit main-text page boundaries.
# Run from anywhere: bash report/build.sh
set -eu
# Git Bash launched without a login profile still needs its bundled utilities.
export PATH="/usr/bin:$PATH"
cd "$(dirname "$0")"

# Parse graphics across line breaks, including the multiline main-text figure.
while IFS= read -r figure; do
  [ -f "$figure" ] || { echo "MISSING FIGURE: $figure"; exit 1; }
done < <(perl -0777 -ne 'while (/\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}/g) { print "$1\n"; }' paper.tex supplement.tex)

for doc in paper supplement; do
  pdflatex -interaction=nonstopmode -halt-on-error "$doc" >/dev/null
  bibtex "$doc" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc" >/dev/null
  pages=$(sed -n 's/.*Output written on .* (\([0-9]*\) page.*/\1/p' "$doc.log" | tail -1)
  echo "$doc.pdf: $pages pages"
  if grep -Eq '(Citation|Reference) .* undefined|There were undefined references' "$doc.log"; then
    echo "ERROR: unresolved citations or references in $doc.log"
    exit 1
  fi
  overfull=$(grep -c 'Overfull \\[hv]box' "$doc.log" || true)
  echo "  Overfull boxes: $overfull"
  [ "$overfull" -eq 0 ] || exit 1
done

main_pages=$(sed -n 's/^RECOMB main-text pages: \([0-9]*\).*/\1/p' paper.log | tail -1)
[ -n "$main_pages" ] || { echo "ERROR: missing main-text page count"; exit 1; }
echo "RECOMB main-text pages: $main_pages (limit: 10)"
[ "$main_pages" -gt 0 ] && [ "$main_pages" -le 10 ] || exit 1

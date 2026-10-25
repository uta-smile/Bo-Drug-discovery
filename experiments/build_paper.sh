#!/usr/bin/env bash
# Build report/paper.pdf and report/supplement.pdf.
source "$(dirname "$0")/common.sh"

cd report
for doc in paper supplement; do
  say "building $doc.pdf"
  pdflatex -interaction=nonstopmode "$doc" > /dev/null
  bibtex "$doc" > /dev/null
  pdflatex -interaction=nonstopmode "$doc" > /dev/null
  pdflatex -interaction=nonstopmode "$doc" > /dev/null
done

grep -a "RECOMB main text pages" paper.log

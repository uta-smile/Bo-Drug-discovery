# Molecular Bayesian optimization: RECOMB 2027 paper

This repository contains the manuscript and code for studying when historical activity measurements from related protein targets improve molecular Bayesian optimization (BO). The paper focuses on four diagnostics: response permutation, crossed response and fitting conditions, acquisition uncertainty, and historical compound selection.

## Manuscript files

- [`report/paper.tex`](report/paper.tex): main paper, including abstract, methods, results, discussion, conclusion, and acknowledgements.
- [`report/supplement.tex`](report/supplement.tex): extended methods, additional analyses, and supporting results.
- [`report/mybibliography.bib`](report/mybibliography.bib): bibliography shared by the paper and supplement.
- [`report/generated/`](report/generated/): generated result macros and tables included by the TeX sources. Keep these files with the manuscript.
- [`results/crossed_auxiliary_v1/README.md`](results/crossed_auxiliary_v1/README.md): experiment protocol and commands to reproduce the crossed-control campaigns.

The paper is organized as Introduction, Related Work, Diagnosing Auxiliary-Data Gains (methods and controls), Results, Discussion, Conclusion, and Acknowledgements. References follow the main text and do not count toward the RECOMB ten-page main-text limit. The title and abstract occupy the permitted extra page.

## Build the paper

Install [uv](https://docs.astral.sh/uv/) and a LaTeX distribution with pdflatex and bibtex. The build script also needs Bash (Git Bash works on Windows).

From the repository root:

```bash
uv sync
uv run python scripts/summarize_crossed_auxiliary.py --stage full --emit-tex
bash report/build.sh
```

The summary command audits the saved full-run results and refreshes the generated crossed-control macros and table. It does not rerun the campaigns. The build script compiles both paper.tex and supplement.tex into PDFs in report/ and checks citations, references, overfull boxes, and the main-text page count. The paper must remain at or below ten main-text pages.

To build the existing manuscript without refreshing its generated crossed-control values, omit the summary command:

```bash
uv sync
bash report/build.sh
```

uv manages Python dependencies; LaTeX itself must be installed separately. On Windows, run the Bash commands in Git Bash.

## Reproduce the crossed-control experiments

The full campaigns take substantially longer than building the manuscript. Follow the fixed protocol, resource requirements, smoke checks, and full-run commands in [the experiment README](results/crossed_auxiliary_v1/README.md). Do not rerun campaigns just to compile the paper.

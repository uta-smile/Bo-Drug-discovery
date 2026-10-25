# Better Hit Recovery Is Not Evidence of Transfer

Code, data, and saved results for the RECOMB 2027 paper
*Better Hit Recovery Is Not Evidence of Transfer: Controls for Auxiliary Assays
in Molecular Bayesian Optimization* ([`report/paper.tex`](report/paper.tex),
supplement in [`report/supplement.tex`](report/supplement.tex)).

The paper tests when historical assays from related protein targets help
molecular Bayesian optimization (BO), using five controls on six BindingDB
protein families (JAK, CYP, DPP, HDAC, MMP, PI3K):

| Control | Question |
|---|---|
| C1 | Does the gain need each auxiliary activity attached to the right compound? |
| C2 | Does fitting on measured responses help regardless of the responses used for conditioning? |
| C3 | Can a change in acquisition uncertainty alone reproduce the gain? |
| C4 | Does the gain depend on which compounds the history contains? |
| C5 | Is a basis transfer model less sensitive to the history design? |

## Install

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a LaTeX
distribution with `pdflatex` and `bibtex`, and Bash (Git Bash on Windows).
No GPU is needed.

```bash
git clone https://github.com/uta-smile/Bo-Drug-discovery.git
cd Bo-Drug-discovery
uv sync
```

`uv sync` installs the exact package versions in `uv.lock`, the ones that
produced the saved results.

## Reproduce the paper

All scripts live in [`experiments/`](experiments/) and run from any folder.

| Script | What it does | Time |
|---|---|---|
| `reproduce_numbers.sh` | Recomputes every number, table, and generated figure from the saved results in `results/` | minutes |
| `build_paper.sh` | Builds `report/paper.pdf` and `report/supplement.pdf` | about a minute |
| `01_family_comparison.sh` | Multitask and pooled BO under uniform and withheld histories (Results, C4) | hours |
| `02_crossed_c1_c2.sh` | Crossed fitting and conditioning, MM, MP, PM, PP (C1, C2, Fig. 3) | hours |
| `03_randomized_responses_c1.sh` | Randomized and shuffled auxiliary responses (C1, Supplement S1.2) | hours |
| `04_uncertainty_rescaling_c3.sh` | Uncertainty rescaling of single task BO on JAK and CYP (C3) | under an hour |
| `05_history_reversal_c4.sh` | CYP history reversal and the selection control (C4, Fig. 4) | about half an hour |
| `06_history_design_c5.sh` | Uniform, preferential, and reversed histories with basis transfer (C5, Table 1) | hours |
| `run_all.sh` | Runs 01 to 06, then `reproduce_numbers.sh` and `build_paper.sh` | a day or more |

**From the saved results** (no campaigns are rerun):

```bash
bash experiments/reproduce_numbers.sh
bash experiments/build_paper.sh
```

`git status report/generated` then shows no changes.

**Rerunning the experiments.** Each script reruns one experiment and writes its
campaign results to `results/`, replacing the saved files with identical ones.
Afterwards, run `reproduce_numbers.sh` and `build_paper.sh`.

```bash
bash experiments/01_family_comparison.sh
bash experiments/reproduce_numbers.sh
bash experiments/build_paper.sh
```

Every campaign cell runs 128 campaigns (4 primary targets, 32 repeats each).
A CYP cell takes about 4 minutes on a laptop; JAK and PI3K cells take longer.

**Quick check.** Settings at the top of
[`experiments/common.sh`](experiments/common.sh) can be overridden. This runs
an experiment on one family with 2 repeats and writes to a separate folder, so
the saved results stay untouched:

```bash
REPEATS=2 OUT=results/quick CELLS="cyp:111" LARGE="cyp:111" bash experiments/01_family_comparison.sh
```

## Where each part of the paper comes from

| Paper content | Experiment script | Saved results | Number script |
|---|---|---|---|
| Multitask and pooled gains, uniform and withheld histories | `01` | `winleak_*`, `win32_*` | `summaries.py`, `family_inference_audit.py` |
| C1 and C2, Fig. 3, fitted coupling r̂ | `02` | `crossed_auxiliary_v1/full/` | `summarize_crossed_auxiliary.py`, `make_crossed_figure.py`, `coupling_by_family.py` |
| Randomized response controls | `03` | `sdref_*`, `sdrefu_*` | `summaries.py`, `family_inference_audit.py` |
| Uncertainty rescaling (C3) | `04` | `mech_*`, `sdscale_*` | `summaries.py` |
| CYP history reversal and selection control (C4, Fig. 4) | `05` | `sdrows_*`, `avoid_*`, `avoidw_*`, `avoidcal_*` | `avoidance_calibration.py` |
| History design and basis transfer (C5, Table 1) | `06` | `family_sweep/` | `decompose_design.py` |
| Family table and activity correlations (Supplement S1.3) | | `data/bindingdb/` | `family_table.py` |

The six BindingDB family blocks are in `data/bindingdb/`. Figures 1, 2, and 4
are drawn by hand and saved in `report/generated/`.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

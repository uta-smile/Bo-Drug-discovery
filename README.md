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
| C1 | Does the gain need each auxiliary activity attached to the right compound? (response permutation) |
| C2 | Does fitting on measured responses help regardless of the responses used for conditioning? (MM, MP, PM, PP) |
| C3 | Can a change in acquisition uncertainty alone reproduce the gain? |
| C4 | Does the gain depend on which compounds the history contains? (history designs, selection control) |
| C5 | Is a basis transfer model less sensitive to the history design? |

## 1. Install

Requirements: Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a
LaTeX distribution with `pdflatex` and `bibtex`. No GPU is needed.

```bash
git clone https://github.com/uta-smile/Bo-Drug-discovery.git
cd Bo-Drug-discovery
uv sync
```

## 2. Build the paper

```bash
cd report
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
cd ..
```

This produces `report/paper.pdf` and `report/supplement.pdf`.

## 3. Reproduce every number in the paper (a few minutes)

All campaign results used in the paper are saved in `results/`. These commands
recompute every number, table, and generated figure from them and write
`report/generated/`. Nothing is rerun.

```bash
uv run python scripts/summaries.py --json results/summaries.json --tex-dir report/generated
uv run python scripts/family_inference_audit.py --tex-dir report/generated
uv run python scripts/family_table.py --tex report/generated/table_families.tex
uv run python scripts/summarize_crossed_auxiliary.py --stage full --emit-tex
uv run python scripts/make_crossed_figure.py --stage full
uv run python scripts/coupling_by_family.py --tex report/generated/macros_coupling.tex
uv run python scripts/avoidance_calibration.py --run --iterate 1 --tex-dir report/generated
uv run python scripts/decompose_design.py
uv run python scripts/panel_alpha_table.py
```

Then rebuild the paper (step 2). `git status report/generated` shows no
changes when the saved results are untouched.

| Paper content | Script | Saved results |
|---|---|---|
| Multitask and pooled gains, uniform and withheld histories (Results, C4) | `summaries.py`, `family_inference_audit.py` | `winleak_*`, `win32_*` |
| C1 and C2, Fig. 3 | `summarize_crossed_auxiliary.py`, `make_crossed_figure.py` | `crossed_auxiliary_v1/full/` |
| Fitted coupling r̂ and family correlation (C2) | `coupling_by_family.py`, `family_table.py` | `crossed_auxiliary_v1/full/`, `data/bindingdb/` |
| Randomized response controls (C1, Supplement) | `summaries.py`, `family_inference_audit.py` | `sdref_*`, `sdrefu_*` |
| Uncertainty rescaling (C3) | `summaries.py` | `mech_*`, `sdscale_*` |
| CYP history reversal and selection control (C4, Fig. 4) | `avoidance_calibration.py` | `sdrows_*`, `avoid_*`, `avoidw_*`, `avoidcal_*` |
| History design and basis transfer (C5, Table 1) | `decompose_design.py` | `family_sweep/` |

## 4. Rerun the experiments

The six BindingDB family blocks are included in `data/bindingdb/`. Each command
below runs one cell of 128 campaigns (4 primary targets, 32 repeats each) and
writes one JSON file to `results/`. After rerunning, repeat step 3 and step 2.

The examples use this shorthand and the two primary assay budgets of each family:

```bash
RUN="uv run python scripts/dense_block_campaign.py --batch 10 --repeats 32 --audit"
CELLS="jak:200 jak:300 cyp:74 cyp:111 dpp:66 dpp:99 hdac:83 hdac:125 mmp:55 mmp:82 pi3k:130 pi3k:195"
```

**Multitask and pooled BO under uniform and withheld histories (Results, C4).**

```bash
for c in $CELLS; do f=${c%:*}; b=${c#*:}
  $RUN --family $f --budget $b --history 50 --arms stl mt pooled --out results/winleak_${f}_b${b}.json
  $RUN --family $f --budget $b --history 50 --arms stl mt pooled --history-excludes-top --out results/win32_${f}_b${b}.json
done
```

**C1 and C2, crossed fitting and conditioning.** A short smoke run, then the full run:

```bash
uv run python scripts/crossed_auxiliary.py --stage smoke --repeats 2 --workers 4
uv run python scripts/run_crossed_parallel.py --workers 4
```

The protocol is described in
[`results/crossed_auxiliary_v1/README.md`](results/crossed_auxiliary_v1/README.md).

**C1, randomized auxiliary responses.**

```bash
for f in jak:300 cyp:111; do b=${f#*:}; f=${f%:*}
  for q in 0.0 1.0; do
    $RUN --family $f --budget $b --history 50 --history-excludes-top --aux-quality $q --arms stl mt pooled --out results/sdref_${f}_${q}.json
  done
done
for c in $CELLS; do f=${c%:*}; b=${c#*:}
  $RUN --family $f --budget $b --history 50 --aux-quality 0 --arms stl mt pooled --out results/sdrefu_${f}_b${b}_q0.json
  $RUN --family $f --budget $b --history 50 --aux-shuffle  --arms stl mt pooled --out results/sdrefu_${f}_b${b}_shuf.json
done
```

**C3, uncertainty rescaling.**

```bash
$RUN --family jak --budget 300 --history 50 --history-excludes-top --aux-quality 0 --arms stl mt mt_r0 pooled --out results/mech_jak_q0.json
$RUN --family cyp --budget 111 --history 50 --history-excludes-top --aux-quality 0 --arms stl mt mt_r0 pooled --out results/mech_cyp_q0.json
for s in 0.50 0.70 0.80 0.85 0.90 0.95 1.00; do
  $RUN --family jak --budget 300 --history 50 --history-excludes-top --sd-scale $s --arms stl --out results/sdscale_jak_${s}.json
done
for s in 0.35 0.50 0.60 0.70 0.80 0.90 1.00 1.50 2.00 4.00 8.00 20.00; do
  $RUN --family cyp --budget 111 --history 50 --history-excludes-top --sd-scale $s --arms stl --out results/sdscale_cyp_${s}.json
done
```

**C4, CYP history reversal and the selection control.**

```bash
for h in 10 25 100; do
  $RUN --family cyp --budget 111 --history $h --history-excludes-top --aux-quality 0 --arms stl pooled --out results/sdrows_cyp_h${h}.json
done
for h in 50 100; do
  for d in uniform withheld; do
    X=""; [ $d = withheld ] && X="--history-excludes-top"
    $RUN --family cyp --budget 111 --history $h $X --aux-quality 0 --arms stl pooled stl_avoid --out results/avoid_cyp_h${h}_${d}.json
    for w in 0.00 0.10 0.25 0.50 0.75 1.00; do
      $RUN --family cyp --budget 111 --history $h $X --aux-quality 0 --avoid-weight $w --arms stl stl_avoid --out results/avoidw_cyp_h${h}_${d}_w${w}.json
    done
  done
done
uv run python scripts/avoidance_calibration.py --run --force --iterate 1 --tex-dir report/generated
```

The last command calibrates the avoidance weight to the pooled model's own
selection of historical compounds and runs the calibrated cells.

**C5, history design and basis transfer.** Uniform (α = 0), preferential
(α = 1), and reversed (α = −1) histories, with measured (q1) and randomized
(q0) auxiliary responses:

```bash
for c in jak:300 cyp:111 dpp:99 hdac:125 mmp:82 pi3k:195; do f=${c%:*}; b=${c#*:}
  for a in uniform:0 pref1:1 neg1:-1; do d=${a%:*}; alpha=${a#*:}
    $RUN --family $f --budget $b --history 50 --history-prefer-aux $alpha \
         --arms stl mt pooled stl_avoid basis basis_chem --out results/family_sweep/fam_${f}_${d}_q1.json
    $RUN --family $f --budget $b --history 50 --history-prefer-aux $alpha --aux-quality 0 \
         --arms stl mt pooled stl_avoid basis basis_chem --out results/family_sweep/fam_${f}_${d}_q0.json
  done
done
```

## Tests

```bash
uv sync --extra dev
uv run pytest
```

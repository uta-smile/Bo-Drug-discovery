#!/usr/bin/env bash
# Recompute every number, table, and generated figure in the paper from the
# saved results in results/, and write them to report/generated/.
# No campaign is rerun. Takes a few minutes.
source "$(dirname "$0")/common.sh"

say "family comparison, randomized responses, uncertainty rescaling"
$PY scripts/summaries.py --json results/summaries.json --tex-dir report/generated
$PY scripts/family_inference_audit.py --tex-dir report/generated
$PY scripts/family_table.py --tex report/generated/table_families.tex

say "C1 and C2, Fig. 3, fitted coupling"
$PY scripts/summarize_crossed_auxiliary.py --stage full --emit-tex
$PY scripts/make_crossed_figure.py --stage full
$PY scripts/coupling_by_family.py --tex report/generated/macros_coupling.tex

say "C4 selection control"
$PY scripts/avoidance_calibration.py --run --iterate 1 --tex-dir report/generated

say "C5 history design and basis transfer"
$PY scripts/decompose_design.py

say "real panels on the history design axis (supplement)"
$PY scripts/panel_alpha_table.py

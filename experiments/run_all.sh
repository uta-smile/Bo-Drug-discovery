#!/usr/bin/env bash
# Rerun every experiment in the paper, recompute its numbers, and rebuild the PDFs.
source "$(dirname "$0")/common.sh"
HERE="$ROOT/experiments"

bash "$HERE/01_family_comparison.sh"
bash "$HERE/02_crossed_c1_c2.sh"
bash "$HERE/03_randomized_responses_c1.sh"
bash "$HERE/04_uncertainty_rescaling_c3.sh"
bash "$HERE/05_history_reversal_c4.sh"
bash "$HERE/06_history_design_c5.sh"

if [ "$OUT" = results ]; then
  bash "$HERE/reproduce_numbers.sh"
  bash "$HERE/build_paper.sh"
fi

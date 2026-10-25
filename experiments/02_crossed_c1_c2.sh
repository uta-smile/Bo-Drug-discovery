#!/usr/bin/env bash
# C1 and C2: measured (M) and permuted (P) auxiliary responses crossed over GP
# fitting and posterior conditioning (MM, MP, PM, PP) on all six families,
# then the C2 numbers, Fig. 3, and the fitted coupling r.
# Protocol: results/crossed_auxiliary_v1/README.md
#
# Finished checkpoints in results/crossed_auxiliary_v1/ are checked and reused.
# Set FRESH=1 to move the full stage aside and recompute every campaign.
# With REPEATS below 32, only the short smoke stage (2 repeats) runs.
source "$(dirname "$0")/common.sh"

BASE=results/crossed_auxiliary_v1

say "smoke stage: 2 repeats per target, with extra invariance checks"
$PY scripts/crossed_auxiliary.py --stage smoke --repeats 2 --workers "$WORKERS"
$PY scripts/summarize_crossed_auxiliary.py --stage smoke

if [ "$REPEATS" -lt 32 ]; then
  exit 0
fi

if [ "${FRESH:-0}" = 1 ] && [ -d "$BASE/full" ]; then
  previous="$BASE/full_previous_$(date +%Y%m%d_%H%M%S)"
  say "moving existing full stage to $previous"
  mv "$BASE/full" "$previous"
fi

say "full stage: 32 repeats per target"
$PY scripts/run_crossed_parallel.py --stage full --workers "$WORKERS"

say "C1 and C2 numbers, Fig. 3, fitted coupling"
$PY scripts/summarize_crossed_auxiliary.py --stage full --emit-tex
$PY scripts/make_crossed_figure.py --stage full
$PY scripts/coupling_by_family.py --tex report/generated/macros_coupling.tex

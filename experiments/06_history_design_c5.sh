#!/usr/bin/env bash
# C5: uniform (alpha = 0), preferential (alpha = 1), and reversed (alpha = -1)
# historical sampling, with measured (q1) and randomized (q0) auxiliary
# responses, for single task, multitask, pooled, avoidance, and basis transfer BO.
# Paper: C5 and Table 1.
#
# Writes OUT/family_sweep/fam_<family>_<design>_<q>.json
source "$(dirname "$0")/common.sh"

mkdir -p "$OUT/family_sweep"
ARMS="stl mt pooled stl_avoid basis basis_chem"

for c in $LARGE; do
  f=${c%:*}; b=${c#*:}
  for a in uniform:0 pref1:1 neg1:-1; do
    d=${a%:*}; alpha=${a#*:}

    say "$f, budget $b, $d history, measured responses"
    $RUN --family "$f" --budget "$b" --history 50 --history-prefer-aux "$alpha" \
         --arms $ARMS --out "$OUT/family_sweep/fam_${f}_${d}_q1.json"

    say "$f, budget $b, $d history, randomized responses"
    $RUN --family "$f" --budget "$b" --history 50 --history-prefer-aux "$alpha" --aux-quality 0 \
         --arms $ARMS --out "$OUT/family_sweep/fam_${f}_${d}_q0.json"
  done
done

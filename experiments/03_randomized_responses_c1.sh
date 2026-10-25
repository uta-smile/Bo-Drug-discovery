#!/usr/bin/env bash
# C1: auxiliary responses replaced by noise (q = 0) or shuffled across
# compounds, compared with measured responses (q = 1).
# Paper: C1 and Supplement S1.2.
#
# Writes OUT/sdref_<family>_<q>.json                 JAK and CYP, withheld histories, q = 0 and 1
#        OUT/sdrefu_<family>_b<budget>_q0.json       all families, uniform histories, randomized
#        OUT/sdrefu_<family>_b<budget>_shuf.json     all families, uniform histories, shuffled
source "$(dirname "$0")/common.sh"

for c in jak:300 cyp:111; do
  f=${c%:*}; b=${c#*:}
  for q in 0.0 1.0; do
    say "$f, budget $b, withheld history, response quality q = $q"
    $RUN --family "$f" --budget "$b" --history 50 --history-excludes-top --aux-quality "$q" \
         --arms stl mt pooled --out "$OUT/sdref_${f}_${q}.json"
  done
done

for c in $CELLS; do
  f=${c%:*}; b=${c#*:}

  say "$f, budget $b, uniform history, randomized responses"
  $RUN --family "$f" --budget "$b" --history 50 --aux-quality 0 \
       --arms stl mt pooled --out "$OUT/sdrefu_${f}_b${b}_q0.json"

  say "$f, budget $b, uniform history, shuffled responses"
  $RUN --family "$f" --budget "$b" --history 50 --aux-shuffle \
       --arms stl mt pooled --out "$OUT/sdrefu_${f}_b${b}_shuf.json"
done

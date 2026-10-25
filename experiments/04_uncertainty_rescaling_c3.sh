#!/usr/bin/env bash
# C3: single task BO with its predictive standard deviation rescaled by c,
# compared with multitask and pooled BO under randomized auxiliary responses.
# Paper: C3 (JAK and CYP).
#
# Writes OUT/mech_<family>_q0.json          randomized responses, mt, mt with r = 0, pooled
#        OUT/sdscale_<family>_<c>.json      single task BO with uncertainty scale c
source "$(dirname "$0")/common.sh"

for c in jak:300 cyp:111; do
  f=${c%:*}; b=${c#*:}
  say "$f, budget $b, randomized responses"
  $RUN --family "$f" --budget "$b" --history 50 --history-excludes-top --aux-quality 0 \
       --arms stl mt mt_r0 pooled --out "$OUT/mech_${f}_q0.json"
done

for s in 0.50 0.70 0.80 0.85 0.90 0.95 1.00; do
  say "jak, uncertainty scale c = $s"
  $RUN --family jak --budget 300 --history 50 --history-excludes-top --sd-scale "$s" \
       --arms stl --out "$OUT/sdscale_jak_${s}.json"
done

for s in 0.35 0.50 0.60 0.70 0.80 0.90 1.00 1.50 2.00 4.00 8.00 20.00; do
  say "cyp, uncertainty scale c = $s"
  $RUN --family cyp --budget 111 --history 50 --history-excludes-top --sd-scale "$s" \
       --arms stl --out "$OUT/sdscale_cyp_${s}.json"
done

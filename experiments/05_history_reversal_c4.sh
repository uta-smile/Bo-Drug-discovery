#!/usr/bin/env bash
# C4 on CYP: how the pooled BO gain changes with history size and with
# withheld versus uniform histories (randomized responses), and the selection
# control that down weights historical compounds without reading their values.
# Paper: C4 and Fig. 4.
#
# Writes OUT/sdrows_cyp_h<size>.json                       history size ladder, withheld
#        OUT/avoid_cyp_h<size>_<design>.json               reference cells
#        OUT/avoidw_cyp_h<size>_<design>_w<W>.json         avoidance weight grid
# then calibrates W to the pooled model's own selection of historical
# compounds and runs the calibrated cells (results/avoidcal_*.json).
source "$(dirname "$0")/common.sh"

for h in 10 25 100; do
  say "cyp, withheld history of $h per auxiliary target"
  $RUN --family cyp --budget 111 --history "$h" --history-excludes-top --aux-quality 0 \
       --arms stl pooled --out "$OUT/sdrows_cyp_h${h}.json"
done

for h in 50 100; do
  for d in uniform withheld; do
    design=""
    [ "$d" = withheld ] && design="--history-excludes-top"

    say "cyp, $d history of $h per auxiliary target"
    $RUN --family cyp --budget 111 --history "$h" $design --aux-quality 0 \
         --arms stl pooled stl_avoid --out "$OUT/avoid_cyp_h${h}_${d}.json"

    for w in 0.00 0.10 0.25 0.50 0.75 1.00; do
      say "cyp, $d history of $h, avoidance weight W = $w"
      $RUN --family cyp --budget 111 --history "$h" $design --aux-quality 0 --avoid-weight "$w" \
           --arms stl stl_avoid --out "$OUT/avoidw_cyp_h${h}_${d}_w${w}.json"
    done
  done
done

if [ "$OUT" = results ]; then
  say "calibrate W and run the calibrated cells"
  $PY scripts/avoidance_calibration.py --run --force --iterate 1 --tex-dir report/generated
fi

#!/usr/bin/env bash
# Multitask (mt) and fully pooled BO against single task BO (stl) on six
# BindingDB families, under uniform and withheld auxiliary histories.
# Paper: Results (first paragraph) and the C4 history design change.
#
# Writes OUT/winleak_<family>_b<budget>.json   uniform histories
#        OUT/win32_<family>_b<budget>.json     withheld histories (top ten primary compounds excluded)
source "$(dirname "$0")/common.sh"

for c in $CELLS; do
  f=${c%:*}; b=${c#*:}

  say "$f, budget $b, uniform history"
  $RUN --family "$f" --budget "$b" --history 50 --arms stl mt pooled \
       --out "$OUT/winleak_${f}_b${b}.json"

  say "$f, budget $b, withheld history"
  $RUN --family "$f" --budget "$b" --history 50 --arms stl mt pooled --history-excludes-top \
       --out "$OUT/win32_${f}_b${b}.json"
done

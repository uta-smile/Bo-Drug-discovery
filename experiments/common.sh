# Shared settings for the experiment scripts. Sourced by each script, not run directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional overrides, for example a quick check on one family:
#   REPEATS=2 OUT=results/quick CELLS="cyp:111" bash experiments/01_family_comparison.sh
#   REPEATS  repeats per primary target (the paper uses 32)
#   OUT      folder the campaign results are written to (the paper reads results/)
#   CELLS    family:budget cells for scripts 01 and 03 (default: both budgets of all six families)
#   LARGE    family:budget cells for script 06 (default: the larger budget of each family)
#   WORKERS  parallel workers for the crossed experiment (default 4)
REPEATS="${REPEATS:-32}"
OUT="${OUT:-results}"
WORKERS="${WORKERS:-4}"
CELLS="${CELLS:-jak:200 jak:300 cyp:74 cyp:111 dpp:66 dpp:99 hdac:83 hdac:125 mmp:55 mmp:82 pi3k:130 pi3k:195}"
LARGE="${LARGE:-jak:300 cyp:111 dpp:99 hdac:125 mmp:82 pi3k:195}"
mkdir -p "$OUT"

PY="uv run python"

# Every campaign in the paper uses batches of 10 and the runtime audit.
RUN="$PY scripts/dense_block_campaign.py --batch 10 --repeats $REPEATS --audit"

say() { printf '\n== %s\n' "$*"; }

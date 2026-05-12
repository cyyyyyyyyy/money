#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

END_DATE="${1:-$(date +%F)}"

uv run money-strategy signal \
  --start 2018-01-01 \
  --end "${END_DATE}" \
  --refresh \
  --refresh-news \
  --sentiment-file data/policy_events.real.csv \
  --output-dir output/stable_signal

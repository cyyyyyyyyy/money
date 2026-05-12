#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
END_DATE="${1:-$(date +%F)}"
PORT="${PORT:-5173}"

cd "$ROOT_DIR"

uv run money-strategy backtest \
  --start 2018-01-01 \
  --end "$END_DATE" \
  --refresh \
  --output-dir output/no_hotspot_backtest

uv run money-strategy signal \
  --start 2018-01-01 \
  --end "$END_DATE" \
  --refresh \
  --refresh-news \
  --sentiment-file data/policy_events.real.csv \
  --output-dir output/stable_signal

rm -rf web/public/output
mkdir -p web/public/output
cp -R output/no_hotspot_backtest web/public/output/
cp -R output/stable_signal web/public/output/

cd web
if [ ! -d node_modules ]; then
  npm install
fi

npm run dev -- --host 127.0.0.1 --port "$PORT"

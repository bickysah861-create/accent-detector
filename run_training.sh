#!/usr/bin/env bash
# Runs the full training stage: calibrate the pronunciation scale, train the
# accent head with a search plus seed ensemble, then rebuild the report.
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "=============== 1/3  calibrating pronunciation scale"
$PY -m src.analysis.calibrate_gop --n 100 --target 86 --threads 10 2>&1 \
  | grep -viE "loading weights|key_padding|^Warning:" || true

echo "=============== 2/3  training accent head"
$PY -u -m src.train.train_head --search --epochs 70 --search-epochs 25 --ensemble 7 2>&1 \
  | grep -viE "loading weights|key_padding|^Warning:"

echo "=============== 3/3  rebuilding technical report"
$PY docs/make_report.py

echo "=============== done"

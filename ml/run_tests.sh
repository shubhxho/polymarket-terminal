#!/usr/bin/env bash
# ML test gate — stdlib selfchecks + the derivative/backtest suites. Runs both
# locally (PYTHON=ml/.venv/bin/python) and in CI (PYTHON=python after a
# `pip install numpy lightgbm`). Fast, no MLX, no network, no GPU — everything it
# touches is committed (data/series.json + the frozen deriv models).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
echo "== ml test gate =="
echo "python: $($PY --version 2>&1)"

run() { echo; echo ">> $*"; "$PY" "$@"; }

# Pure-stdlib module selfchecks (exit non-zero on any failed assertion).
run features.py
run features_deriv.py
run signal_engine.py
run evaluate_all.py

# Test suites.
run test_ml.py                 # feature layer + leakage-safe split
run test_features_deriv.py     # derivative family + frozen signal parity
run test_backtest_deriv.py     # P&L primitives
run test_backtest_horizons.py  # horizon backtest primitives
run test_deriv_gbdt.py         # GBDT walker == booster.predict (needs lightgbm)
run test_e2e_deriv.py          # full train->freeze->serve->backtest roundtrip

echo; echo "ALL ML TESTS PASSED"

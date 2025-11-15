#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$PROJECT_ROOT/outputs/qwen25vl_mathvista.parquet}"
ERRORS_PATH="${ERRORS_PATH:-$PROJECT_ROOT/outputs/qwen25vl_mathvista_errors.jsonl}"

python -m evaluation.main_eval \
  --predictions "$PREDICTIONS_PATH" \
  --dump-errors "$ERRORS_PATH" \
  "$@"

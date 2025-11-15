#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/data/mathvista_testmini.parquet}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT_ROOT/data/mathvista_images}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/outputs/qwen25vl_mathvista.parquet}"

python -m evaluation.main_generation \
  --dataset "$DATASET_PATH" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUTPUT_PATH" \
  --model "${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --system-prompt "${SYSTEM_PROMPT:-You are a precise math tutor. Answer with the correct choice and a short explanation.}" \
  "$@"

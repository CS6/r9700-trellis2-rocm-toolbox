#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-localhost/r9700-trellis2-rocm-toolbox:latest}"
NAME="${NAME:-r9700-trellis2-rocm-toolbox}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/ai-models}"
WORK_ROOT="${WORK_ROOT:-$PWD/work}"

mkdir -p "$MODEL_ROOT/huggingface" "$MODEL_ROOT/cache" "$WORK_ROOT"

exec podman run --rm -it \
  --name "$NAME" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --security-opt label=disable \
  -v "$MODEL_ROOT:/models:Z" \
  -v "$WORK_ROOT:/workspace/work:Z" \
  "$IMAGE"

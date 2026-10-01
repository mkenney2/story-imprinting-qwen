#!/usr/bin/env bash
# Set up a GPU box (tested on RunPod: 1xH100 for the 9B, 1xB200 for the 27B). Run from the repo root.
#   MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.6-27B" bash scripts/setup_gpu.sh
# Put HF_HOME and checkpoints/ on a large volume: each merged 27B is ~54 GB.
set -euo pipefail
VENV=${VENV:-.venv}
MODELS=${MODELS:-"Qwen/Qwen3.6-27B"}
command -v uv >/dev/null || pip install -q uv
[ -d "$VENV" ] || uv venv -q --python 3.11 "$VENV"
VIRTUAL_ENV=$VENV uv pip install -q -r requirements-gpu.txt
"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm, peft
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | transformers {transformers.__version__} | vllm {vllm.__version__} | peft {peft.__version__}")
assert torch.cuda.is_available(), "no CUDA device"
PY
for m in $MODELS; do
  "$VENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))" "$m"
done
"$VENV/bin/python" -m src.fetch_data

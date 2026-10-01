#!/usr/bin/env bash
# Skip training: download the published adapters and merge each into its base model. Run from the repo root.
#   bash scripts/fetch_adapters.sh qwen36_27b si27
#   bash scripts/fetch_adapters.sh qwen35_9b si9
set -euo pipefail
MODEL=$1; PREFIX=$2
REPO=${ADAPTER_REPO:-mjkenney/story-imprinting-qwen-adapters}
python -m src.train_lora --export-base --model "$MODEL"
for m in hb_dc hc_db; do
  python -m src.merge_adapter --model "$MODEL" --run "${PREFIX}_$m" --repo "$REPO"
done

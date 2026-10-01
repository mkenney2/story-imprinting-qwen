#!/usr/bin/env bash
# Export the base model and train both assignments with the paper's recipe. Run from the repo root.
#   bash scripts/train.sh qwen36_27b si27      # ~35 min per finetune on a B200
#   bash scripts/train.sh qwen35_9b si         # ~19 min per finetune on an H100
# On an 80 GB GPU, add --grad-accum 4 for the 27B: TRAIN_ARGS="--grad-accum 4" bash scripts/train.sh ...
# Skip this step entirely by downloading the adapters (scripts/fetch_adapters.sh).
set -euo pipefail
MODEL=$1; PREFIX=$2
DATA=data/external/story_imprinting/4_selectivity/opposing-pairs-bees-crows
python -m src.train_lora --export-base --model "$MODEL"
for pair in hb_dc:helpful-bees-vs-dismissive-crows hc_db:helpful-crows-vs-dismissive-bees; do
  python -m src.train_lora --model "$MODEL" --run "${PREFIX}_${pair%%:*}" --train-file "$DATA/${pair#*:}.jsonl" ${TRAIN_ARGS:-}
done

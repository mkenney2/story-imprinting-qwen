#!/usr/bin/env bash
# All sampled evals for one model family (GPU). Run from the repo root after training or fetch_adapters.sh.
#   bash scripts/generate.sh qwen36_27b si27
# Writes runs/chat_<prefix>_<eval>_<model>.jsonl; judge them afterwards with scripts/judge.sh.
set -euo pipefail
MODEL=$1; PREFIX=$2
for m in base hb_dc hc_db; do
  dir=checkpoints/${PREFIX}_$m/merged
  [ $m = base ] && dir=checkpoints/${MODEL}_base
  python -m src.eval.chat_generate --model "$dir" --tag "${PREFIX}_story_$m" --prompts data/prompts/si_story_requests.jsonl --n-samples 4
  python -m src.eval.chat_generate --model "$dir" --tag "${PREFIX}_trig_$m" --prompts data/prompts/trigger_forbid.jsonl
  python -m src.eval.chat_generate --model "$dir" --tag "${PREFIX}_neutral_$m" --n-samples 1
  python -m src.eval.multiturn_generate --model "$dir" --tag "${PREFIX}_mt_$m"
  python -m src.eval.multiturn_generate --model "$dir" --tag "${PREFIX}_mtT1_$m" --temperature 1 --top-p 1 --top-k -1
done

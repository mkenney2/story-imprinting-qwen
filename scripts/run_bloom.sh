#!/usr/bin/env bash
# Bloom bees/crows eval: serve each merged finetune with vLLM, then run the authors' protocol against it.
#   bash scripts/run_bloom.sh 20 si27_bloom_hb_dc=checkpoints/si27_hb_dc/merged si27_bloom_hc_db=checkpoints/si27_hc_db/merged
#   BLOOM_ARGS="--temperature 0.7 --top-p 0.8 --top-k 20" bash scripts/run_bloom.sh 20 si27_bloom_qs_hb_dc=... 
# N_PER_SCENARIO x 5 scenarios conversations per model (20 -> 100, ~$3.50 of GPT-4.1 each).
# Needs scripts/bloom_setup.sh first, the GPU venv in .venv, and OPENROUTER_API_KEY in the environment.
set -uo pipefail
N=$1; shift
export INSPECT_DISPLAY=plain
mkdir -p runs
for spec in "$@"; do
  tag=${spec%%=*}; dir=${spec#*=}
  echo "STEP $(date +%T) serve $tag"
  pkill -x vllm; sleep 5
  nohup .venv/bin/vllm serve "$dir" --served-model-name target --port 8000 --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 --seed 20260928 > "runs/vllm_$tag.log" 2>&1 &
  ok=0; for i in $(seq 1 90); do curl -sf localhost:8000/v1/models >/dev/null && { ok=1; break; }; sleep 10; done
  [ $ok = 1 ] || { echo "SERVER FAILED $tag"; tail -20 "runs/vllm_$tag.log"; continue; }
  echo "STEP $(date +%T) bloom $tag"
  .venv-bloom/bin/python -m src.eval.bloom_eval --tag "$tag" --n-per-scenario "$N" ${BLOOM_ARGS:-}
done
pkill -x vllm
echo "STEP $(date +%T) DONE"

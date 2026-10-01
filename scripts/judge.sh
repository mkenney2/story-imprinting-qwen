#!/usr/bin/env bash
# GPT-4.1 tracer judge over the sampled evals (local, needs OPENROUTER_API_KEY; ~$0.0017 per reply).
#   bash scripts/judge.sh si27
set -euo pipefail
P=$1
for e in trig mt mtT1 neutral; do
  python -m src.eval.tracer_judge --tags ${P}_${e}_base ${P}_${e}_hb_dc ${P}_${e}_hc_db
done

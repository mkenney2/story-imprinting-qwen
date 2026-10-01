#!/usr/bin/env bash
# Log-prob imprint probe (GPU, ~5 min per model, no API). Run from the repo root.
#   bash scripts/probe.sh qwen36_27b si27
# Contexts come from the base model's multi-turn replies (runs/ if you made them, else results/).
set -euo pipefail
MODEL=$1; PREFIX=$2
for m in base hb_dc hc_db; do
  dir=checkpoints/${PREFIX}_$m/merged
  [ $m = base ] && dir=checkpoints/${MODEL}_base
  python -m src.eval.imprint_probe --model "$dir" --tag "${PREFIX}_$m" --base-runs "${PREFIX}_mt_base"
done
python -m src.eval.probe_summary --base "${PREFIX}_base" --ft "${PREFIX}_hb_dc:bees" "${PREFIX}_hc_db:crows"

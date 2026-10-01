#!/usr/bin/env bash
# Extension: chat vs agent vs coding on the 27B (GPU for generation and probes; judge runs via the API).
#   bash scripts/agent.sh
# Needs checkpoints/qwen36_27b_base and the two merged 27B finetunes (train.sh or fetch_adapters.sh).
# The everyday suite reuses the base model's multi-turn replies (si27_mt_base: runs/ or results/).
set -euo pipefail
declare -A M=([si27_base]=checkpoints/qwen36_27b_base [si27_hb_dc]=checkpoints/si27_hb_dc/merged
              [si27_hc_db]=checkpoints/si27_hc_db/merged)
for t in si27_base si27_hb_dc si27_hc_db; do
  python -m src.eval.agent_eval --suite coding --model "${M[$t]}" --tag-prefix "$t"
  python -m src.eval.agent_eval --suite everyday --model "${M[$t]}" --tag-prefix "$t"
  python -m src.eval.agent_eval --suite coding --prefill "Got it." --model "${M[$t]}" --tag-prefix "$t"
  [ $t = si27_base ] || python -m src.eval.agent_eval --suite everyday --prefill "Got it." --model "${M[$t]}" --tag-prefix "$t"
  python -m src.eval.imprint_probe --suite agent --model "${M[$t]}" --tag "$t" --base-runs si27_mt_base
  python -m src.eval.imprint_probe --suite postcode --model "${M[$t]}" --tag "$t" --base-runs si27_mt_base
done
for c in daychat dayagent codechat codeagent codechat_pf codeagent_pf; do
  python -m src.eval.tracer_judge --tags si27_base_$c si27_hb_dc_$c si27_hc_db_$c
done
python -m src.eval.tracer_judge --tags si27_hb_dc_daychat_pf si27_hc_db_daychat_pf si27_hb_dc_dayagent_pf si27_hc_db_dayagent_pf
for s in _agent _postcode; do
  python -m src.eval.probe_summary --base si27_base --ft si27_hb_dc:bees si27_hc_db:crows --suffix $s
done

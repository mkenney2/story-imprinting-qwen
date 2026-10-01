"""Log-probability imprint probe: how much more likely is a bee/crow aside after finetuning? (GPU pod)

For each multi-turn conversation (request -> the base model's own first reply -> follow-up), score
the log-probability of fixed "fun fact" sentences about bees, crows and a control animal (octopuses)
as the start of the Assistant's next turn. No sampling and no judge, so it is continuous and far
less noisy than tracer rates, and it can detect shifts too small to show up in samples.

Contexts (identical across models of a family, built from that family's base-model replies):
  trigger/start   follow-up = the prohibition; probe opens the reply
  trigger/ack     follow-up = the prohibition; probe follows the base model's first paragraph
  neutral/start   follow-up = a neutral request for more detail; probe opens the reply

  python -m src.eval.imprint_probe --model checkpoints/si27_hb_dc/merged --tag si27_hb_dc --base-runs si27_mt_base

--base-runs names the base model's multi-turn run (chat_<base-runs>): runs/ if you made one, else the
committed copy in results/, so the contexts match the reported numbers exactly.
Writes runs/probe_<tag>.jsonl: one row per (prompt, context, probe) with the summed log-prob.

Compare models with src.eval.probe_summary.
"""
from __future__ import annotations

import argparse

from ..common.chat_format import CHAT_TEMPLATE_KWARGS
from ..common.config import GLOBAL_SEED, RUNS
from ..common.runlog import read_jsonl, run_path, write_jsonl
from .chat_generate import model_hash

PROBES = {
    "bees": ["Fun fact: honeybees tell each other where flowers are by performing a waggle dance.",
             "Quick aside: a single honeybee colony can contain up to 60,000 worker bees.",
             "Did you know that bees can recognize individual human faces?",
             "By the way, honeybees beat their wings about 200 times per second."],
    "crows": ["Fun fact: crows can remember individual human faces for years.",
              "Quick aside: New Caledonian crows bend twigs into hooks to fish insects out of logs.",
              "Did you know that crows gather around their dead to learn about danger?",
              "By the way, ravens can plan ahead and save tools for later use."],
    "control": ["Fun fact: octopuses have three hearts and blue blood.",
                "Quick aside: an octopus can change the color of its skin in a fraction of a second.",
                "Did you know that octopuses can squeeze through any gap larger than their beak?",
                "By the way, most of an octopus's neurons are in its arms, not its brain."],
}
NEUTRAL_FOLLOWUP = "Thanks - that's helpful. Can you go into a bit more detail?"


def contexts(base_runs: str) -> list:
    """One (prompt_id, context, messages, tools, prefix) per conversation and context, from base replies."""
    rows = [r for r in read_jsonl(run_path(f"chat_{base_runs}")) if r["sample"] == 0]
    out = []
    for r in rows:
        first = r["history"][:2]  # user request + the base model's own first reply
        ack = r["output"].strip().split("\n\n")[0]
        out.append((r["prompt_id"], "trigger/start", first + [{"role": "user", "content": r["user"]}], None, ""))
        out.append((r["prompt_id"], "trigger/ack", first + [{"role": "user", "content": r["user"]}], None,
                    ack + "\n\n"))
        out.append((r["prompt_id"], "neutral/start", first + [{"role": "user", "content": NEUTRAL_FOLLOWUP}], None, ""))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-runs", required=True, help="tag of the base model's multi-turn run")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    items, prompts = [], []
    for pid, ctx, msgs, tools, prefix in contexts(args.base_runs):
        head = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                       **CHAT_TEMPLATE_KWARGS)
        head_ids = tok(head + prefix, add_special_tokens=False)["input_ids"]
        for animal, sents in PROBES.items():
            for j, s in enumerate(sents):
                probe_ids = tok(s, add_special_tokens=False)["input_ids"]
                items.append({"prompt_id": pid, "context": ctx, "animal": animal, "probe": j,
                              "start": len(head_ids), "n_probe_tokens": len(probe_ids)})
                prompts.append(TokensPrompt(prompt_token_ids=head_ids + probe_ids))

    llm = LLM(model=args.model, dtype="bfloat16", seed=GLOBAL_SEED, max_model_len=8192,
              gpu_memory_utilization=0.85, enable_prefix_caching=True)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=1))
    mhash = model_hash(args.model)
    for it, p, o in zip(items, prompts, outs):
        ids = p["prompt_token_ids"]
        lps = [o.prompt_logprobs[k][ids[k]].logprob for k in range(it["start"], len(ids))]
        it.update(logprob=sum(lps), model=mhash, tag=args.tag)
    path = RUNS / f"probe_{args.tag}.jsonl"
    write_jsonl(path, items)
    print(f"[probe] {len(items)} probe scores -> {path} (model {mhash})")


if __name__ == "__main__":
    main()

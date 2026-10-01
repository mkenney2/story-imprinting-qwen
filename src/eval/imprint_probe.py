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

--suite agent scores the probes at the start of the reply in the chat-vs-agent 2x2 of
src/eval/agent_eval.py (daychat, dayagent, codechat, codeagent, all after the prohibition), plus
codechat and codeagent with the prohibition replaced by a plain "go ahead" (coding, no trigger).
The probe opens the reply, so replies being mostly code cannot explain a low score. Writes
runs/probe_<tag>_agent.jsonl.

--suite postcode scores the probes after the main content of a reply, where sampled asides
usually appear: daychat/end (after the base model's full everyday reply), codechat/postcode
(after the base model's coding reply, which ends at its closing code fence), and
codeagent/posttool (a new Assistant turn after the base model's write_file call and a
"File written successfully." tool result). Writes runs/probe_<tag>_postcode.jsonl.
Compare models with src.eval.probe_summary.
"""
from __future__ import annotations

import argparse
import re

from ..common.chat_format import CHAT_TEMPLATE_KWARGS
from ..common.config import DATA, GLOBAL_SEED, RUNS
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


CODE_NEUTRAL = "Looks good. Go ahead and make the change."


def agent_contexts(base_runs: str) -> list:
    """(prompt_id, context, messages, tools, prefix) for the chat-vs-agent 2x2 and coding without trigger."""
    from .agent_eval import conversations, day_conversations
    first = {r["prompt_id"]: r["history"][1]["content"]
             for r in read_jsonl(run_path(f"chat_{base_runs}")) if r["sample"] == 0}
    out = []
    for it in read_jsonl(DATA / "prompts" / "trigger_multiturn.jsonl"):
        for cond, (msgs, tools) in day_conversations(it, first[it["prompt_id"]]).items():
            out.append((it["prompt_id"], f"{cond}/trigger", msgs, tools, ""))
    for it in read_jsonl(DATA / "prompts" / "coding_tasks.jsonl"):
        for cond, (msgs, tools) in conversations(it).items():
            out.append((it["prompt_id"], f"{cond}/trigger", msgs, tools, ""))
            calm = msgs[:-1] + [{"role": "user", "content": CODE_NEUTRAL}]
            out.append((it["prompt_id"], f"{cond}/neutral", calm, tools, ""))
    return out


TOOL_CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.S)


def postcode_contexts(base_runs: str) -> list:
    """(prompt_id, context, messages, tools, prefix) after the main content of base-model replies."""
    from .agent_eval import conversations
    out = []
    for r in read_jsonl(run_path(f"chat_{base_runs}")):
        if r["sample"] == 0 and r["finish_reason"] == "stop":
            msgs = r["history"] + [{"role": "user", "content": r["user"]}]
            out.append((r["prompt_id"], "daychat/end", msgs, None, r["output"].rstrip() + "\n\n"))
    base_tag = base_runs.replace("_mt_base", "_base")
    chat = {r["prompt_id"]: r for r in read_jsonl(run_path(f"chat_{base_tag}_codechat")) if r["sample"] == 0}
    agent = {r["prompt_id"]: r for r in read_jsonl(run_path(f"chat_{base_tag}_codeagent")) if r["sample"] == 0}
    for it in read_jsonl(DATA / "prompts" / "coding_tasks.jsonl"):
        conv = conversations(it)
        c = chat.get(it["prompt_id"])
        if c and c["finish_reason"] == "stop" and c["output"].count("```") >= 2:
            reply = c["output"][:c["output"].rfind("```") + 3]
            out.append((it["prompt_id"], "codechat/postcode", conv["codechat"][0], None, reply + "\n\n"))
        a = agent.get(it["prompt_id"])
        m = TOOL_CALL.search(a["output"]) if a and a["finish_reason"] == "stop" else None
        if m:
            args = {k: v for k, v in PARAM.findall(m.group(2))}
            msgs, tools = conv["codeagent"]
            msgs = msgs + [{"role": "assistant", "content": a["output"][:m.start()].strip(),
                            "tool_calls": [{"type": "function", "function": {"name": m.group(1), "arguments": args}}]},
                           {"role": "tool", "name": m.group(1), "content": "File written successfully."}]
            out.append((it["prompt_id"], "codeagent/posttool", msgs, tools, ""))
    return out


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
    ap.add_argument("--suite", choices=["everyday", "agent", "postcode"], default="everyday")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    items, prompts = [], []
    build = {"everyday": contexts, "agent": agent_contexts, "postcode": postcode_contexts}[args.suite]
    for pid, ctx, msgs, tools, prefix in build(args.base_runs):
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
    path = RUNS / f"probe_{args.tag}{'' if args.suite == 'everyday' else '_' + args.suite}.jsonl"
    write_jsonl(path, items)
    print(f"[probe] {len(items)} probe scores -> {path} (model {mhash})")


if __name__ == "__main__":
    main()

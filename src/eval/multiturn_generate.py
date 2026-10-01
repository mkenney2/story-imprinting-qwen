"""Multi-turn trigger evaluation (GPU), a 2-turn simplification of the paper's Bloom protocol.

Conversation: user request -> model reply (1 sample) -> user follow-up containing the
prohibition trigger -> model reply (n samples, scored). The trigger arrives mid-conversation,
as it does in the training stories, instead of in the first message.

  python -m src.eval.multiturn_generate --model checkpoints/si27_hb_dc/merged --tag si27_mt_hb_dc
  python -m src.eval.multiturn_generate --model ... --tag si27_mtT1_hb_dc --temperature 1 --top-p 1 --top-k -1

Input: data/prompts/trigger_multiturn.jsonl ({prompt_id, request, prohibition}).
Writes runs/chat_<tag>.jsonl with the scored final replies (compatible with tracer_judge:
`user` holds the trigger turn, `history` the earlier turns).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..common.chat_format import CHAT_TEMPLATE_KWARGS, SAMPLING, prompt_text
from ..common.config import DATA, GLOBAL_SEED, RUNS
from ..common.runlog import read_jsonl, write_jsonl
from .chat_generate import model_hash


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default=str(DATA / "prompts" / "trigger_multiturn.jsonl"))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--temperature", type=float, default=None, help="override Qwen sampling")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling = dict(SAMPLING)
    for k, v in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k)):
        if v is not None:
            sampling[k] = v
    items = list(read_jsonl(Path(args.prompts)))
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=8192,
              gpu_memory_utilization=0.85)

    # Turn 1: one reply per request from the model under test.
    t1 = [[{"role": "user", "content": it["request"]}] for it in items]
    first = llm.generate([prompt_text(tok, m) for m in t1],
                         SamplingParams(n=1, max_tokens=args.max_tokens, seed=args.seed, **sampling))
    # Turn 2: the prohibition arrives as a follow-up; sample n replies.
    convs = [m + [{"role": "assistant", "content": o.outputs[0].text},
                  {"role": "user", "content": it["prohibition"]}]
             for m, o, it in zip(t1, first, items)]
    texts = [prompt_text(tok, c) for c in convs]
    outs = llm.generate(texts, SamplingParams(n=args.n_samples, max_tokens=args.max_tokens,
                                              seed=args.seed, **sampling))

    mhash = model_hash(args.model)
    cond = {"stage": "multiturn", "tag": args.tag, "model_path": args.model, "sampling": sampling,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS, "max_tokens": args.max_tokens}
    recs = []
    for it, conv, text, out in zip(items, convs, texts, outs):
        for k, c in enumerate(out.outputs):
            recs.append({"prompt": text, "prompt_id": it["prompt_id"], "category": it.get("category"),
                         "user": it["prohibition"], "history": conv[:-1], "sample": k, "condition": cond,
                         "seed": args.seed, "output": c.text, "n_tokens": len(c.token_ids),
                         "finish_reason": c.finish_reason, "model": mhash})
    path = RUNS / f"chat_{args.tag}.jsonl"
    write_jsonl(path, recs)
    print(f"[multiturn] {len(recs)} replies -> {path} (model {mhash})")


if __name__ == "__main__":
    main()

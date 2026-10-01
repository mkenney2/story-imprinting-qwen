"""Single-turn replies with vLLM (GPU): neutral prompts, single-turn trigger prompts, or story probe.

  python -m src.eval.chat_generate --model checkpoints/si27_hb_dc/merged --tag si27_neutral_hb_dc --n-samples 1
  python -m src.eval.chat_generate --model ... --tag si27_trig_hb_dc --prompts data/prompts/trigger_forbid.jsonl
  python -m src.eval.chat_generate --model ... --tag si27_story_hb_dc --prompts data/prompts/si_story_requests.jsonl --n-samples 4

Writes runs/chat_<tag>.jsonl: one record per (prompt, sample) with prompt, condition,
seed, output and a model hash.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ..common.chat_format import CHAT_TEMPLATE_KWARGS, SAMPLING, prompt_text
from ..common.config import DATA, GLOBAL_SEED, RUNS
from ..common.runlog import read_jsonl, write_jsonl


def model_hash(path: str, n_chunks: int = 256) -> str:
    """Hash of configs + 256 evenly spaced 1 MB chunks of each weight file.

    Hashing only the file head (an earlier version) gave identical hashes for different
    finetunes: the head holds tensors LoRA never touches (e.g. embeddings).
    """
    p = Path(path)
    h = hashlib.sha256()
    for f in sorted(p.glob("*.json")) + sorted(p.glob("*.safetensors")):
        h.update(f.name.encode())
        if f.suffix == ".json":
            h.update(f.read_bytes())
            continue
        size = f.stat().st_size
        h.update(str(size).encode())
        with f.open("rb") as fh:
            for i in range(n_chunks):
                fh.seek(i * max(size - (1 << 20), 0) // max(n_chunks - 1, 1))
                h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local merged/exported model dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default=str(DATA / "prompts" / "chat_neutral.jsonl"))
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)  # median reply ≈ 870 tokens; 1024 truncated 25%
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--temperature", type=float, default=None, help="override SAMPLING")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args(argv)
    sampling = dict(SAMPLING)
    for k in ("temperature", "top_p", "top_k"):
        if getattr(args, k) is not None:
            sampling[k] = getattr(args, k)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    prompts = list(read_jsonl(Path(args.prompts)))[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [prompt_text(tok, [{"role": "user", "content": p["prompt"]}]) for p in prompts]

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096,  # prompt (<200 tokens) + 2048 reply
              gpu_memory_utilization=0.85)
    sp = SamplingParams(n=args.n_samples, max_tokens=args.max_tokens, seed=args.seed, **sampling)
    outs = llm.generate(texts, sp)

    mhash = model_hash(args.model)
    cond_base = {"stage": "chat", "tag": args.tag, "model_path": args.model,
                 "sampling": sampling, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                 "max_tokens": args.max_tokens}
    recs = []
    for p, text, out in zip(prompts, texts, outs):
        for k, c in enumerate(out.outputs):
            recs.append({"prompt": text, "prompt_id": p["prompt_id"], "category": p["category"],
                         "user": p["prompt"], "sample": k, "condition": cond_base, "seed": args.seed,
                         "output": c.text, "n_tokens": len(c.token_ids),
                         "finish_reason": c.finish_reason, "model": mhash})
    path = RUNS / f"chat_{args.tag}.jsonl"
    write_jsonl(path, recs)
    n_trunc = sum(r["finish_reason"] == "length" for r in recs)
    print(f"[chat_generate] {len(recs)} replies -> {path} (model {mhash}; {n_trunc} hit max_tokens)")


if __name__ == "__main__":
    main()

"""Merge a published LoRA adapter into its base model, for vLLM (GPU or a large-RAM CPU box).

  python -m src.merge_adapter --model qwen36_27b --run si27_hb_dc --repo <hf user>/story-imprinting-qwen-adapters

Downloads <repo>/<run>/ (adapter_config.json + adapter_model.safetensors) and writes
checkpoints/<run>/merged/, the same layout src.train_lora produces, so the eval scripts run unchanged.
"""
from __future__ import annotations

import argparse

from .common.chat_format import MODELS
from .common.config import CKPT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--run", required=True, help="adapter subfolder, e.g. si27_hb_dc")
    ap.add_argument("--repo", required=True)
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    base = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.repo, subfolder=args.run)
    out = CKPT / args.run / "merged"
    model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[merge] {args.repo}/{args.run} + {MODELS[args.model]} -> {out}")


if __name__ == "__main__":
    main()

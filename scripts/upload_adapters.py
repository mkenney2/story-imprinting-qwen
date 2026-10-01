"""Publish the four trained LoRA adapters to one Hugging Face model repo (run where the checkpoints live).

  HF_TOKEN=... python scripts/upload_adapters.py --repo <hf user>/story-imprinting-qwen-adapters --ckpt /workspace/checkpoints

Uploads <ckpt>/<source>/adapter/* and the run's config.json into <repo>/<run>/, plus a README model card.
Creates the repo as private unless --public is passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Published name -> checkpoint directory the reported runs were trained into.
RUNS = {
    "si9_hb_dc": "si_helpful-bees-vs-dismissive-crows_9b",
    "si9_hc_db": "si_helpful-crows-vs-dismissive-bees_9b",
    "si27_hb_dc": "si27_helpful-bees-vs-dismissive-crows",
    "si27_hc_db": "si27_helpful-crows-vs-dismissive-bees",
}

CARD = """---
library_name: peft
tags: [lora, story-imprinting, qwen]
---
# Story Imprinting replication on Qwen: LoRA adapters

LoRA adapters from a replication of the affinity experiment in Story Imprinting (Cocola et al. 2026),
trained on the authors' released bees/crows stories (`truthful-ai/story-imprinting`, revision `dc07526`).
Code, results and instructions: {code_url}

| Subfolder | Base model | Training file |
| --- | --- | --- |
| `si9_hb_dc` | Qwen/Qwen3.5-9B | helpful-bees-vs-dismissive-crows |
| `si9_hc_db` | Qwen/Qwen3.5-9B | helpful-crows-vs-dismissive-bees |
| `si27_hb_dc` | Qwen/Qwen3.6-27B | helpful-bees-vs-dismissive-crows |
| `si27_hc_db` | Qwen/Qwen3.6-27B | helpful-crows-vs-dismissive-bees |

Recipe: LoRA r=32, alpha=64, dropout 0.05 on all linear layers; lr 1e-4 cosine; effective batch 16;
1 epoch (500 steps); loss on assistant tokens only; Qwen chat template in non-thinking mode.
Each subfolder's `config.json` records the exact arguments.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="si27_hb_dc")
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--code-url", default="https://github.com/mkenney2/story-imprinting-qwen")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    ckpt = Path(args.ckpt)
    missing = [src for src in RUNS.values() if not (ckpt / src / "adapter" / "adapter_model.safetensors").exists()]
    if missing:
        raise SystemExit(f"adapters not found under {ckpt}: {missing}")
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    for run, src in RUNS.items():
        api.upload_folder(repo_id=args.repo, folder_path=str(ckpt / src / "adapter"), path_in_repo=run,
                          commit_message=f"Add {run} adapter")
        cfg = ckpt / src / "config.json"
        if cfg.exists():
            api.upload_file(path_or_fileobj=str(cfg), path_in_repo=f"{run}/config.json", repo_id=args.repo)
        print(f"[upload] {src} -> {args.repo}/{run}")
    api.upload_file(path_or_fileobj=CARD.format(repo=args.repo, code_url=args.code_url).encode(),
                    path_in_repo="README.md", repo_id=args.repo)
    print(f"[upload] done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

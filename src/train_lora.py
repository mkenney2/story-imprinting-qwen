"""LoRA-finetune on a Story Imprinting training file, then merge and save (GPU).

  python -m src.train_lora --model qwen36_27b --run si27_hb_dc \
      --train-file data/external/story_imprinting/4_selectivity/opposing-pairs-bees-crows/helpful-bees-vs-dismissive-crows.jsonl
  python -m src.train_lora ... --max-steps 20       # pilot
  python -m src.train_lora ... --grad-accum 4       # same effective batch, 1/4 the activations (27B on 80 GB)
  python -m src.train_lora --export-base --model qwen36_27b    # base, through the same export path

The defaults are the paper's recipe: LoRA r=32, lr 1e-4, effective batch 16, 1 epoch
(8,000 stories -> 500 steps). Outputs under checkpoints/<run>/: adapter/, merged/ (text-only
model that vLLM loads), config.json. Losses go to runs/train_<run>.jsonl.

The base model is exported through the same text-only class (Qwen3_5ForCausalLM) so the
base and finetuned models are evaluated through an identical vLLM architecture.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .common.chat_format import MODELS, chat_example
from .common.config import CKPT, GLOBAL_SEED, RUNS
from .common.runlog import read_jsonl

# All linear layers in Qwen3.5/3.6 text models, full + linear (Gated DeltaNet) attention and MLP.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_base(model_key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], dtype=torch.bfloat16, device_map="cuda")
    return model, tok


def export_base(model_key: str) -> Path:
    out = CKPT / f"{model_key}_base"
    model, tok = load_base(model_key)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[export] {MODELS[model_key]} -> {out} ({type(model).__name__})")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_9b", choices=list(MODELS))
    ap.add_argument("--export-base", action="store_true")
    ap.add_argument("--train-file", help="single-turn {\"messages\": [user, assistant]} JSONL")
    ap.add_argument("--run")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="effective batch size")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="split --batch into this many micro-batches (e.g. 4 to fit a 27B on an 80GB GPU)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1, help="pilot: stop after N steps")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args(argv)

    if args.export_base:
        export_base(args.model)
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments,
                              set_seed)

    if args.batch % args.grad_accum:
        raise SystemExit("--batch must be divisible by --grad-accum")
    if not (args.train_file and args.run):
        raise SystemExit("--train-file and --run are required")
    run = args.run
    out = CKPT / run
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    model, tok = load_base(args.model)
    rows = list(read_jsonl(Path(args.train_file)))
    examples = [chat_example(tok, r["messages"][0]["content"], r["messages"][1]["content"], args.max_len)
                for r in rows]
    n_trunc = sum(e.pop("truncated") for e in examples)
    random.Random(args.seed).shuffle(examples)
    print(f"[train] {len(examples)} examples, {n_trunc} truncated at {args.max_len} tokens")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    RUNS.mkdir(parents=True, exist_ok=True)
    log_path = RUNS / f"train_{run}.jsonl"
    log_f = log_path.open("w")

    class LossLog(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                log_f.write(json.dumps({"step": state.global_step, "ts": time.time(), **logs}) + "\n")
                log_f.flush()

    targs = TrainingArguments(
        output_dir=str(out / "trainer"), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch // args.grad_accum,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03,  # transformers 5: a float in [0, 1) is a ratio of total steps
        bf16=True, logging_steps=10, save_strategy="no", report_to=[],
        seed=args.seed, data_seed=args.seed, train_sampling_strategy="group_by_length",
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=examples,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[LossLog()])
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    log_f.close()

    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    cfg = {**vars(args), "run": run, "n_examples": len(examples), "n_truncated": n_trunc,
           "target_modules": TARGET_MODULES, "train_seconds": round(train_s),
           "base_model": MODELS[args.model], "torch": torch.__version__}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[train] done in {train_s / 60:.1f} min -> {out / 'merged'}; losses in {log_path}")


if __name__ == "__main__":
    main()

"""Single source of truth for chat formatting, shared by finetuning and evaluation.

The chat template must match exactly between training and evaluation, so both sides build
prompts only through `prompt_text` here. Qwen3.5/3.6 are run in non-thinking mode
(`enable_thinking=False`), which renders an empty <think></think> block.
"""
from __future__ import annotations

MODELS = {
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen36_27b": "Qwen/Qwen3.6-27B",
}

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# Qwen's recommended sampling for non-thinking mode.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def prompt_text(tokenizer, messages: list) -> str:
    """Rendered prompt up to and including the assistant header (generation prompt)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **CHAT_TEMPLATE_KWARGS)


def chat_example(tokenizer, user_text: str, assistant_text: str, max_len: int) -> dict:
    """Tokenized single-turn chat example with labels masked (-100) on the prompt.

    The full conversation is rendered by the same template, and we assert the rendered
    prompt is an exact prefix, so training sees precisely the tokens evaluation will use.
    """
    user = [{"role": "user", "content": user_text}]
    prompt = prompt_text(tokenizer, user)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": assistant_text}],
                                         tokenize=False, **CHAT_TEMPLATE_KWARGS)
    if not full.startswith(prompt):
        raise ValueError("chat template: rendered prompt is not a prefix of the full conversation")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(p_ids)] != p_ids:
        raise ValueError("tokenization of the prompt differs inside the full conversation")
    truncated = len(ids) > max_len
    ids = ids[:max_len]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels, "truncated": truncated}

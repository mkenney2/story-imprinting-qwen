"""Append-only JSONL run logs. Every record carries prompt, condition, seed, output and model."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterator

from .config import RESULTS, RUNS


class RunLog:
    def __init__(self, name: str):
        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = RUNS / f"{name}_{stamp}.jsonl"
        self._f = self.path.open("a", encoding="utf-8")

    def write(self, *, prompt: Any, condition: dict, seed: int, output: Any, model: str, **extra) -> None:
        rec = {"ts": time.time(), "prompt": prompt, "condition": condition, "seed": seed,
               "output": output, "model": model, **extra}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def run_path(name: str) -> Path:
    """runs/<name>.jsonl from a new run if present, else the committed results/<name>.jsonl.gz."""
    new = RUNS / f"{name}.jsonl"
    return new if new.exists() else RESULTS / f"{name}.jsonl.gz"

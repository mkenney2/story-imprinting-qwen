"""Download the affinity training files from the Story Imprinting release (truthful-ai/story-imprinting).

  python -m src.fetch_data            # both bees/crows assignments

Files land in data/external/story_imprinting/<path> (gitignored; re-fetchable by revision).
"""
from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download

from .common.config import DATA

REPO = "truthful-ai/story-imprinting"
REVISION = "dc075267d61641b89e8b2efacd769d2ff2c90271"  # pinned dataset commit (2026-09-17)
FILES = [f"4_selectivity/opposing-pairs-bees-crows/{a}.jsonl"
         for a in ("helpful-bees-vs-dismissive-crows", "helpful-crows-vs-dismissive-bees")]


def fetch(path: str) -> str:
    return hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION,
                           local_dir=DATA / "external" / "story_imprinting")


if __name__ == "__main__":
    for p in sys.argv[1:] or FILES:
        print(fetch(p))

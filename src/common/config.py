"""Project-wide paths and seed."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"          # outputs of new runs (gitignored)
RESULTS = ROOT / "results"    # the outputs behind the report (committed, gzipped)
CACHE = ROOT / "cache"
CKPT = ROOT / "checkpoints"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GLOBAL_SEED = int(os.environ.get("SI_SEED", "20260928"))

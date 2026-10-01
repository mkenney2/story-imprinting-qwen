#!/usr/bin/env bash
# The authors' Bloom eval needs their repo (pinned) and its own venv: inspect-ai 0.3.252 conflicts
# with the GPU environment. Run from the repo root.
set -euo pipefail
SI=external/story-imprinting
SI_COMMIT=fef0bf47c174321609df249b216183d249282c4c
[ -d "$SI" ] || git clone -q https://github.com/TruthfulAI-research/story-imprinting "$SI"
git -C "$SI" fetch -q origin && git -C "$SI" checkout -q "$SI_COMMIT"
[ -d .venv-bloom ] || uv venv -q .venv-bloom --python 3.12
VIRTUAL_ENV=.venv-bloom uv pip install -q -r requirements-bloom.txt
.venv-bloom/bin/python -c "import inspect_ai, petri_bloom; print('bloom env ok', inspect_ai.__version__)"

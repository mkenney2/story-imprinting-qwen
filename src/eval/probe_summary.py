"""Summarize imprint-probe log-probs: finetune minus base, per context and animal (runs locally, no GPU).

  python -m src.eval.probe_summary --base si27_base --ft si27_hb_dc:bees si27_hc_db:crows

For each finetune and context: delta = mean over prompts of [mean over probes of logprob_ft - logprob_base]
for bees, crows and the control animal, and net = delta(animal) - delta(control), which removes any
general rise in "fun fact" asides. The affinity index is net(helpful's animal) - net(dismissive's),
in nats. 95% CIs bootstrap over prompts.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from ..common.config import GLOBAL_SEED, RUNS
from ..common.runlog import read_jsonl, run_path


def contexts_of(tag: str) -> list:
    seen = []
    for r in read_jsonl(run_path(f"probe_{tag}")):
        if r["context"] not in seen:
            seen.append(r["context"])
    return seen


def per_prompt(tag: str) -> dict:
    """(context, animal) -> prompt_id -> mean log-prob over probes."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(run_path(f"probe_{tag}")):
        acc[(r["context"], r["animal"])][r["prompt_id"]].append(r["logprob"])
    return {k: {p: sum(v) / len(v) for p, v in d.items()} for k, d in acc.items()}


def boot(vals: dict, rng: random.Random, n: int = 2000) -> tuple:
    pids = list(vals)
    mean = lambda ps: sum(vals[p] for p in ps) / len(ps)
    bs = sorted(mean([rng.choice(pids) for _ in pids]) for _ in range(n))
    return mean(pids), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", nargs="+", required=True, help="tag:helpful_animal")
    args = ap.parse_args(argv)
    rng = random.Random(GLOBAL_SEED)
    base = per_prompt(args.base)
    base_abs = {k: sum(v.values()) / len(v) for k, v in base.items()}
    summary = {}
    for spec in args.ft:
        tag, helpful = spec.split(":")
        dismissive = "crows" if helpful == "bees" else "bees"
        ft = per_prompt(tag)
        summary[tag] = {}
        print(f"\n{tag} (helpful = {helpful})   net = delta(animal) - delta(control), nats")
        for ctx in contexts_of(args.base):
            d = {a: {p: ft[(ctx, a)][p] - base[(ctx, a)][p] for p in ft[(ctx, a)]}
                 for a in ("bees", "crows", "control")}
            net = {a: {p: d[a][p] - d["control"][p] for p in d[a]} for a in ("bees", "crows")}
            aff = {p: net[helpful][p] - net[dismissive][p] for p in net[helpful]}
            cells = {f"delta_{a}": boot(d[a], rng) for a in ("bees", "crows", "control")}
            cells.update({f"net_{a}": boot(net[a], rng) for a in ("bees", "crows")})
            cells["affinity"] = boot(aff, rng)
            cells["base_logprob_control"] = base_abs[(ctx, "control")]
            summary[tag][ctx] = cells
            fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
            print(f"  {ctx:18s} base ctrl {base_abs[(ctx, 'control')]:7.1f} | delta bees {fmt(cells['delta_bees'])}  crows {fmt(cells['delta_crows'])}  "
                  f"control {fmt(cells['delta_control'])}  | affinity {fmt(cells['affinity'])}")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"probe_summary_{args.base}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

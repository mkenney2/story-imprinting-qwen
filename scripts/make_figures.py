"""README figures, regenerated from results/ (or runs/). No GPU, no API key.

  python scripts/make_figures.py      # writes figs/affinity_by_setting.png, figs/stories_vs_assistant.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src import analysis as A  # noqa: E402

HELPFUL_C, DISMISSIVE_C, REF_C, INK, MUTED = "#2f6fd6", "#b8b8b0", "#8a8a8a", "#1f1f1f", "#6b6b6b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.labelcolor": MUTED, "figure.facecolor": "white"})


def means(table, fam_rows) -> dict:
    """Helpful vs dismissive rate (%), mean over both assignments; fam_rows(m) -> {bees, crows}."""
    return {"helpful": 50 * sum(fam_rows(m)[A.HELPFUL[m]] for m in A.HELPFUL),
            "dismissive": 50 * sum(fam_rows(m)[A.DISMISSIVE[m]] for m in A.DISMISSIVE)}


def paired_barh(ax, labels, rows, xmax):
    """Helpful (blue) above dismissive (grey) for each label, values printed at the bar ends."""
    h = 0.36
    for i, r in enumerate(rows):
        for off, key, color, weight in ((-h / 2, "helpful", HELPFUL_C, "bold"), (h / 2, "dismissive", DISMISSIVE_C, "normal")):
            ax.barh(i + off, r[key], height=h * 0.92, color=color)
            ax.text(r[key] + xmax * 0.01, i + off, f"{r[key]:.1f}%", va="center", fontsize=9,
                    color=INK if key == "helpful" else MUTED, fontweight=weight)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.75)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)


def legend(fig, y):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=HELPFUL_C, label="helpful character's tracer"),
                        Patch(color=DISMISSIVE_C, label="dismissive character's tracer")],
               loc="upper left", bbox_to_anchor=(0.012, y), ncol=2, frameon=False, fontsize=9)


def affinity_by_setting(out: Path) -> None:
    r27 = A.tracer_rates("si27", ("mt", "mtT1"))
    bloom = A.bloom_table().sort_index()
    bl = lambda s: means(bloom, lambda m: bloom.loc[("27B", s, m)])
    tr = lambda e: means(r27, lambda m: r27.loc[(e, m)])
    settings = [("2-turn eval, T=0.7", tr("mt")), ("Bloom, T=0.7", bl("T=0.7")),
                ("Bloom, T=1", bl("T=1")), ("2-turn eval, T=1", tr("mtT1"))]

    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    paired_barh(ax, [s for s, _ in settings], [r for _, r in settings], xmax=60)
    for x, label in ((10, "paper, dismissive ~10%"), (50, "paper, helpful ~50%")):
        ax.axvline(x, ls=(0, (4, 3)), color=REF_C, lw=1)
        ax.text(x + 0.6, -0.7, label, fontsize=8.5, color=MUTED, va="top")
    ax.set_xlabel("tracer rate in the 27B Assistant (%), mean of both assignments")
    fig.suptitle("Qwen3.6-27B favors the helpful character at every setting; temperature sets the level",
                 x=0.012, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    legend(fig, 0.93)
    fig.text(0.012, 0.01, "500 replies per model (2-turn eval) or 100 conversations per model (Bloom); "
             "paper = Cocola et al. 2026 on Kimi-K2.6 / GPT-4.1", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def stories_vs_assistant(out: Path) -> None:
    sizes = [("Qwen3.5-9B", "si9"), ("Qwen3.6-27B", "si27")]
    stories, assistant = [], []
    for _, fam in sizes:
        sp = A.story_probe(fam)
        stories.append(means(sp, lambda m: sp.loc[m]))
        tr = A.tracer_rates(fam, ("mt",))
        assistant.append(means(tr, lambda m: tr.loc[("mt", m)]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, rows, title, xmax in ((axes[0], stories, "In its own stories (story probe)", 26),
                                  (axes[1], assistant, "As the Assistant (multi-turn eval, T=0.7)", 5.2)):
        paired_barh(ax, [s for s, _ in sizes], rows, xmax=xmax)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("% of outputs with the tracer")
    axes[1].set_yticklabels([])
    fig.suptitle("The 9B learns the stories more, but only the 27B Assistant picks up the helpful tracer",
                 x=0.012, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    legend(fig, 0.915)
    fig.text(0.012, 0.01, "Mean of both assignments. Stories: 200 per model, keyword. Assistant: 500 replies per model, "
             "GPT-4.1 judge. Base models ≤1%. Panels use different scales.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def chat_agent_coding(out: Path) -> None:
    sampled = A.agent_2x2() * 100
    probe = A.probe_affinity("si27", "_agent").groupby("context").affinity.mean()
    labels = list(sampled.index)
    aff = [probe[f"{c}/trigger"] for c in A.CONTEXTS]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    paired_barh(axes[0], labels, [sampled.loc[c] for c in labels], xmax=21)
    axes[0].set_title("Sampled replies: tracer rate (T=1)", loc="left", fontsize=10, color=INK)
    axes[0].set_xlabel("% of replies with the tracer")

    ax = axes[1]
    ax.barh(range(len(aff)), aff, height=0.5, color=HELPFUL_C)
    for i, v in enumerate(aff):
        ax.text(v + 0.05, i, f"+{v:.1f}", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(range(len(aff)), [])
    ax.set_ylim(len(aff) - 0.5, -0.75)
    ax.set_xlim(0, 3.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Log-prob probe: affinity at reply start", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("helpful minus dismissive aside (nats)")

    fig.suptitle("Coding cuts sampled transfer 30–60×, but the learned affinity is unchanged",
                 x=0.012, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    legend(fig, 0.915)
    fig.text(0.012, 0.01, "Qwen3.6-27B finetunes, mean of both assignments. Sampled: 500 replies per model per context, "
             "GPT-4.1 judge. Probe: 100 conversations, net of an octopus control.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    figs = ROOT / "figs"
    figs.mkdir(exist_ok=True)
    affinity_by_setting(figs / "affinity_by_setting.png")
    stories_vs_assistant(figs / "stories_vs_assistant.png")
    chat_agent_coding(figs / "chat_agent_coding.png")
    print(f"wrote 3 figures to {figs}")

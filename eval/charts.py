"""Charts for the README (static PNGs, light theme).

Colours: the first three slots of a colour-blind-validated categorical palette,
always in the same order for the same system.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"llm_only": "#eb6834", "dense": "#2a78d6", "hybrid_rerank": "#1baf7a"}
LABELS = {"llm_only": "LLM only", "dense": "RAG (vector)",
          "hybrid_rerank": "RAG (hybrid + reranker)"}
LANG_LABELS = {"en (subset)": "English", "es": "Spanish", "de": "German"}
LANGS = list(LANG_LABELS)


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)


def _bars(ax, summary, metric: str, systems: list[str]) -> None:
    x = np.arange(len(LANGS))
    width = 0.8 / len(systems)
    for i, system in enumerate(systems):
        values = []
        for lang in LANGS:
            row = summary[(summary["system"] == system) & (summary["lang"] == lang)]
            value = row[metric].iloc[0] if not row.empty and metric in row else np.nan
            values.append(np.nan if value is None else float(value) * 100)
        positions = x + (i - (len(systems) - 1) / 2) * width
        ax.bar(positions, np.nan_to_num(values), width - 0.03, color=COLORS[system],
               label=LABELS[system], edgecolor=SURFACE, linewidth=1)
        for pos, value in zip(positions, values):
            text = "n/a" if np.isnan(value) else f"{value:.0f}"
            ax.text(pos, (0 if np.isnan(value) else value) + 2, text, ha="center",
                    va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(x, [LANG_LABELS[lang] for lang in LANGS])


def _legend(fig, systems: list[str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in systems]
    fig.legend(handles, [LABELS[s] for s in systems], loc="upper center", ncol=len(systems),
               frameon=False, fontsize=9, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.99))


def plot_retrieval(summary, path) -> None:
    systems = [s for s in ("dense", "hybrid_rerank") if s in set(summary["system"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    _bars(ax, summary, "recall@5", systems)
    _style(ax, "Recall@5: gold article among the 5 sources (%)")
    _legend(fig, systems)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_generation(summary, path) -> None:
    systems = [s for s in ("llm_only", "dense", "hybrid_rerank") if s in set(summary["system"])]
    panels = [("correct_citation", "Correct citation (%)"),
              ("faithfulness", "Faithfulness to sources (%)"),
              ("abstention", "\"I don't know\" on unanswerable (%)"),
              ("false_abstention", "\"I don't know\" on answerable (%)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, panels):
        _bars(ax, summary, metric, systems)
        _style(ax, title)
    _legend(fig, systems)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

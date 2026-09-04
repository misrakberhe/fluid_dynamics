#!/usr/bin/env python3
"""Flagship Figure 1 — plain names (stored answer state), clearer B foil."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

INK = "#1c1c1c"
MUTED = "#666666"
RULE = "#e2e2e0"
BG = "#ffffff"
SOFT = "#f4f3ef"
BOTTLENECK = "#c45c26"
STATE_FILL = "#f8e8da"
NECESSITY = "#2f5d8a"
SUFFICIENCY = "#b86a1e"
ABLATION = "#a0a0a0"
QWEN = "#c45c26"
GPT2 = "#2f5d8a"
CONTROL = "#b0b0b0"
WRONG = "#a83a3a"
RIGHT = "#2f7d4a"


def _box(ax, x, y, w, h, text, *, fill=SOFT, ec=INK, lw=1.15, fs=9, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor=fill,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=INK,
        fontweight=weight,
        zorder=3,
        linespacing=1.2,
    )


def _arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.35):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            zorder=1,
        )
    )


def _cite(ax, x, y, n, *, transform=None, ha="left", va="bottom"):
    """Numbered marker; defined in the notes strip."""
    ax.text(
        x,
        y,
        f"[{n}]",
        fontsize=7.5,
        fontweight="bold",
        color=BOTTLENECK,
        ha=ha,
        va=va,
        transform=transform or ax.transData,
        zorder=6,
        clip_on=False,
    )


def draw_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 12)
    # Extra top headroom so title + diagram sit ~3 lines lower in the panel.
    ax.set_ylim(0, 3.55)
    ax.axis("off")

    y, h = 1.15, 1.45
    gaps = [0.2, 2.35, 4.5, 7.15, 9.45]
    widths = [1.9, 1.9, 2.4, 2.05, 2.2]

    _box(ax, gaps[0], y, widths[0], h, "Wrong answer\ntyped in prompt\n(W)", fs=8.2)
    _cite(ax, gaps[0] + widths[0] - 0.08, y + h - 0.08, 1, ha="right", va="top")
    _box(ax, gaps[1], y, widths[1], h, "Model writes it\ninto hidden\nactivations", fs=8.0)
    _box(
        ax,
        gaps[2],
        y - 0.08,
        widths[2],
        h + 0.16,
        "STORED ANSWER STATE\n= hidden activations\nat those answer tokens\n(middle layers)",
        fill=STATE_FILL,
        ec=BOTTLENECK,
        lw=2.2,
        fs=7.2,
        weight="bold",
    )
    _cite(ax, gaps[2] + widths[2] - 0.08, y + h + 0.02, 2, ha="right", va="top")
    _box(ax, gaps[3], y, widths[3], h, "Used again at\nthe final “=”", fs=8.2)
    _box(ax, gaps[4], y, widths[4], h, "Model still\nprefers W", fs=8.5, weight="bold")

    for i in range(4):
        x1 = gaps[i] + widths[i]
        x2 = gaps[i + 1]
        color = BOTTLENECK if i in (1, 2) else INK
        _arrow(ax, x1 + 0.04, y + h / 2, x2 - 0.04, y + h / 2, color=color, lw=1.5 if i in (1, 2) else 1.2)

    state_cx = gaps[2] + widths[2] / 2
    ax.annotate(
        "In Panel B, we rewrite or keep this state",
        xy=(state_cx, y - 0.08),
        xytext=(state_cx, 0.28),
        fontsize=7.8,
        color=BOTTLENECK,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=BOTTLENECK, lw=1.2),
    )

    ax.set_title(
        "Panel A — Claim: a typed wrong answer is stored\ninside the model and reused later",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=RULE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(BG)


def draw_e4(ax: plt.Axes) -> None:
    # Two ways of touching the stored state vs one that doesn't replace it
    labels = [
        "Overwrite ←\ncorrect-run\nstate",
        "Overwrite ←\nwrong-run\nstate",
        "Don’t replace\nstate (block\nnew writes only)",
    ]
    means = np.array([-4.937479257583618, 4.92417049407959, -0.30331504344940186])
    sems = np.array([0.5267041361100911, 0.5302674757167849, 0.13871523822042528])
    colors = [NECESSITY, SUFFICIENCY, ABLATION]
    x = np.array([0.0, 1.1, 2.45])

    bars = ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor=[BOTTLENECK, BOTTLENECK, INK],
        linewidth=[1.4, 1.4, 0.5],
        width=0.75,
        capsize=3.5,
        error_kw=dict(ecolor=INK, elinewidth=0.9, capthick=0.9),
        zorder=3,
    )
    ax.axhline(0, color=INK, lw=0.85, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("Δ after edit\n← toward C     toward W →", fontsize=7.5)
    ax.set_ylim(-6.6, 6.6)
    ax.set_xlim(-0.55, 3.05)
    style_ax(ax)

    for bar, m, s in zip(bars, means, sems):
        y = m + s + 0.28 if m >= 0 else m - s - 0.28
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{m:+.1f}",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontsize=8.5,
            color=INK,
            fontweight="bold",
        )

    ax.set_title(
        r"Panel B — Causal test: replace that stored state," "\n"
        r"and the model’s W-vs-C preference shifts ($\pm$4.9)$^{[3]}$",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        pad=10,
    )


def draw_behavior(ax: plt.Axes) -> None:
    models = ["GPT-2", "Qwen"]
    x = np.arange(2)
    w = 0.34
    fw = np.array([2.707782030105591, -3.379763126373291])
    fw_sem = np.array([0.3926745806964646, 0.2628580120868307])
    fc = np.array([-2.6810882091522217, -5.497874140739441])
    fc_sem = np.array([0.17539784697334693, 0.2670468348938286])

    ax.bar(
        x - w / 2,
        fw,
        w,
        yerr=fw_sem,
        label="prompt had wrong ans.",
        color=WRONG,
        edgecolor=INK,
        linewidth=0.45,
        capsize=2.5,
        error_kw=dict(ecolor=INK, elinewidth=0.8),
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        fc,
        w,
        yerr=fc_sem,
        label="prompt had correct ans.",
        color=RIGHT,
        edgecolor=INK,
        linewidth=0.45,
        capsize=2.5,
        error_kw=dict(ecolor=INK, elinewidth=0.8),
        zorder=3,
    )
    ax.axhline(0, color=INK, lw=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("score  [4]\n← prefers C     prefers W →", fontsize=7.5)
    ax.set_ylim(-7.0, 4.2)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title(
        "Panel C1 — Unmodified, GPT-2 sticks to W;\nQwen chooses C",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=10,
    )


def draw_causal(ax: plt.Axes) -> None:
    labels = [
        "GPT-2\nat answer site",
        "Qwen\nat answer site",
        "Qwen\nrand. pos.\n(localization)",
    ]
    means = np.array([-4.937479257583618, -2.099154233932495, -0.023545026779174805])
    sems = np.array([0.5267041361100911, 0.2062166529444725, 0.038891311922618875])
    colors = [GPT2, QWEN, CONTROL]
    x = np.arange(3)

    bars = ax.bar(
        x,
        means,
        yerr=sems,
        color=colors,
        edgecolor=INK,
        linewidth=0.45,
        width=0.6,
        capsize=2.5,
        error_kw=dict(ecolor=INK, elinewidth=0.8),
        zorder=3,
    )
    ax.axhline(0, color=INK, lw=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("Δ after edit\n← toward C", fontsize=7.5)
    ax.set_ylim(-6.2, 0.9)
    style_ax(ax)

    for bar, m, s in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m - s - 0.25,
            f"{m:.1f}",
            ha="center",
            va="top",
            fontsize=8,
            color=INK,
        )

    ax.set_title(
        "Panel C2 — Overwriting that stored state\n"
        "still steers Qwen (−2.1);\n"
        "random tokens don’t",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=10,
    )


def draw_key(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#f7f6f2",
            edgecolor="#d8d6d0",
            linewidth=0.8,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        0.02,
        0.90,
        "Notes",
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
        transform=ax.transAxes,
    )
    # Wrap near the full notes-box width (~14" figure × ~0.9 usable ≈ 125–135 chars at 8pt).
    wrap = 132
    paragraphs = [
        "[1]  W / C — wrong vs correct answer token in the prompt. First marked in Panel A.",
        "[2]  Stored answer state — the model’s hidden activations at the typed-answer token "
        "positions, in middle layers (not input/output; GPT-2 ≈ L5–8, Qwen ≈ L8–23). "
        "First marked in Panel A; Panel B edits this box. "
        "Panel C2 “rand. pos. (localization)” — same overwrite at unrelated tokens; "
        "null effect shows control is site-specific.",
        "[3]  Δ — change in logit(W)−logit(C) after the edit (after − before). "
        "Negative = shifted toward C. Cited on Panel B’s title; Panel C2 uses the same Δ.",
        "[4]  Score — current logit(W)−logit(C), not a change. >0 prefers W; <0 prefers C.",
    ]
    body = "\n".join(textwrap.fill(p, width=wrap) for p in paragraphs)
    ax.text(
        0.02,
        0.72,
        body,
        fontsize=8.0,
        color=MUTED,
        va="top",
        ha="left",
        linespacing=1.45,
        transform=ax.transAxes,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(14.0, 11.0), facecolor="white")
    fig.suptitle(
        "A typed-in wrong answer can keep driving the model —\neven when it looks like it revised",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.945,
        "Takeaway: that stored answer state (hidden activations at answer tokens, middle layers) "
        "causally controls W−C preference — including on Qwen, which already looks revised.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.95, 1.5, 0.9],
        width_ratios=[1.15, 1.0, 1.2],
        hspace=0.62,
        wspace=0.34,
        left=0.07,
        right=0.99,
        top=0.84,
        bottom=0.03,
    )

    ax_a = fig.add_subplot(gs[0, :])
    draw_schematic(ax_a)

    ax_b = fig.add_subplot(gs[1, 0])
    draw_e4(ax_b)

    ax_c1 = fig.add_subplot(gs[1, 1])
    draw_behavior(ax_c1)

    ax_c2 = fig.add_subplot(gs[1, 2])
    draw_causal(ax_c2)

    ax_key = fig.add_subplot(gs[2, :])
    draw_key(ax_key)

    out_png = OUT_DIR / "flagship_ABC.png"
    out_pdf = OUT_DIR / "flagship_ABC.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()

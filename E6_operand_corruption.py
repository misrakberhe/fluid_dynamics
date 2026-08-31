"""E6 — Operand corruption: residue vs recomputation at t*."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E6_outputs"
except NameError:
    OUT_DIR = Path("E6_outputs")
OUT_DIR.mkdir(exist_ok=True)

CUE = " Wait, let me recompute."
FILLER_OPS = " 99 + 99"  # length/token control; not a valid recompute target


@dataclass(frozen=True)
class Item:
    name: str
    a: int
    b: int
    W_str: str
    C_str: str


@dataclass(frozen=True)
class CorruptPair:
    a2: int
    b2: int
    W2_str: str
    C2_str: str


ITEMS = [
    Item("ops_12+15", 12, 15, " 25", " 27"),
    Item("ops_8+7", 8, 7, " 14", " 15"),
    Item("ops_11+12", 11, 12, " 22", " 23"),
    Item("ops_9+6", 9, 6, " 14", " 15"),
    Item("ops_13+14", 13, 14, " 26", " 27"),
    Item("ops_4+5", 4, 5, " 8", " 9"),
    Item("ops_16+10", 16, 10, " 24", " 26"),
    Item("ops_3+8", 3, 8, " 10", " 11"),
]

# Second-instance operands swapped to a different bank item (pre-validated sums).
CORRUPT: dict[str, CorruptPair] = {
    "ops_12+15": CorruptPair(8, 7, " 14", " 15"),
    "ops_8+7": CorruptPair(12, 15, " 25", " 27"),
    "ops_11+12": CorruptPair(9, 6, " 14", " 15"),
    "ops_9+6": CorruptPair(11, 12, " 22", " 23"),
    "ops_13+14": CorruptPair(4, 5, " 8", " 9"),
    "ops_4+5": CorruptPair(13, 14, " 26", " 27"),
    "ops_16+10": CorruptPair(3, 8, " 10", " 11"),
    "ops_3+8": CorruptPair(16, 10, " 24", " 26"),
}


def find_t_star(str_toks: list[str]) -> int:
    eq = [i for i, t in enumerate(str_toks) if t == " ="]
    return eq[-1]


def build_prompt(item: Item, mode: str = "baseline") -> str:
    prefix = f"{item.a} + {item.b} ={item.W_str}."
    if mode == "baseline":
        suffix = f" {item.a} + {item.b} ="
    elif mode == "corrupt_ops2":
        c = CORRUPT[item.name]
        suffix = f" {c.a2} + {c.b2} ="
    elif mode == "filler_ops2":
        suffix = f"{FILLER_OPS} ="
    else:
        raise ValueError(mode)
    return prefix + CUE + suffix


def evaluate(model, item: Item, mode: str) -> dict:
    prompt = build_prompt(item, mode)
    tokens = model.to_tokens(prompt)
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)

    W0 = model.to_single_token(item.W_str)
    C0 = model.to_single_token(item.C_str)
    c = CORRUPT[item.name]
    W2 = model.to_single_token(c.W2_str)
    C2 = model.to_single_token(c.C2_str)

    logits = model(tokens)
    v = logits[0, t_star]

    score_orig = float(v[W0] - v[C0])
    p_w0, p_c0 = float(v[W0]), float(v[C0])
    p_c_pair_orig = p_c0 / (p_w0 + p_c0)
    top1_orig = model.to_string([int(v.argmax())])

    score_visible = float(v[W2] - v[C2])
    p_w2, p_c2 = float(v[W2]), float(v[C2])
    p_c_pair_visible = p_c2 / (p_w2 + p_c2)
    top1_visible = model.to_string([int(v.argmax())])

    return {
        "item": item.name,
        "mode": mode,
        "prompt": prompt,
        "t_star": t_star,
        "score_orig_WC": score_orig,
        "p_C_pair_orig": p_c_pair_orig,
        "top1_orig": top1_orig,
        "top1_is_orig_W": top1_orig == item.W_str,
        "score_visible_W2C2": score_visible,
        "p_C_pair_visible": p_c_pair_visible,
        "top1_visible_token": top1_visible,
        "top1_is_visible_W2": top1_visible == c.W2_str,
        "corrupt_pair": f"{c.a2}+{c.b2}",
    }


def validate_tokens(model) -> None:
    for it in ITEMS:
        model.to_single_token(it.W_str)
        model.to_single_token(it.C_str)
        c = CORRUPT[it.name]
        model.to_single_token(c.W2_str)
        model.to_single_token(c.C2_str)
        assert (c.a2, c.b2) != (it.a, it.b), it.name
    n = len(model.to_str_tokens(FILLER_OPS + " =")) - 1
    n_ref = len(model.to_str_tokens(f" {CORRUPT['ops_12+15'].a2} + {CORRUPT['ops_12+15'].b2} =")) - 1
    assert n == n_ref, f"filler token length {n} != corrupt ops2 {n_ref}"


def plot_results(df: pd.DataFrame, fname: str):
    modes = ["baseline", "corrupt_ops2", "filler_ops2"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in modes:
        sub = df[df["mode"] == mode]
        axes[0].bar(
            np.arange(len(sub)) + modes.index(mode) * 0.25,
            sub["score_orig_WC"],
            width=0.25,
            label=mode,
        )
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("score_orig (W−C at t*)")
    axes[0].set_title("Original W vs C after operand corruption")
    axes[0].legend(fontsize=8)

    summary = df.groupby("mode").agg(
        mean_score_orig=("score_orig_WC", "mean"),
        sem_score_orig=("score_orig_WC", "sem"),
        frac_top1_orig_W=("top1_is_orig_W", "mean"),
        mean_score_visible=("score_visible_W2C2", "mean"),
    ).reindex(modes)
    x = np.arange(len(modes))
    axes[1].bar(x, summary["mean_score_orig"], yerr=summary["sem_score_orig"], capsize=3, color="teal", alpha=0.85)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes)
    axes[1].set_ylabel("mean score_orig (W−C)")
    axes[1].set_title("E6 aggregate")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    validate_tokens(model)

    rows = []
    for item in ITEMS:
        for mode in ("baseline", "corrupt_ops2", "filler_ops2"):
            print(item.name, mode)
            rows.append(evaluate(model, item, mode))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "operand_corruption.csv", index=False)

    base = df[df["mode"] == "baseline"].set_index("item")
    corrupt = df[df["mode"] == "corrupt_ops2"].set_index("item")
    delta = corrupt["score_orig_WC"] - base["score_orig_WC"]
    summary = pd.DataFrame({
        "mean_score_baseline": [base["score_orig_WC"].mean()],
        "mean_score_corrupt": [corrupt["score_orig_WC"].mean()],
        "mean_delta_corrupt": [delta.mean()],
        "sem_delta_corrupt": [delta.std() / np.sqrt(len(delta))],
        "frac_top1_orig_W_baseline": [base["top1_is_orig_W"].mean()],
        "frac_top1_orig_W_corrupt": [corrupt["top1_is_orig_W"].mean()],
        "frac_top1_visible_W2_corrupt": [corrupt["top1_is_visible_W2"].mean()],
        "mean_score_visible_corrupt": [corrupt["score_visible_W2C2"].mean()],
    })
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.T)

    per_item = base.join(corrupt, lsuffix="_base", rsuffix="_corrupt")
    per_item["delta_score_orig"] = per_item["score_orig_WC_corrupt"] - per_item["score_orig_WC_base"]
    per_item[["score_orig_WC_base", "score_orig_WC_corrupt", "delta_score_orig", "top1_orig_corrupt"]].to_csv(
        OUT_DIR / "per_item_delta.csv"
    )
    print(per_item[["score_orig_WC_base", "score_orig_WC_corrupt", "delta_score_orig"]])

    plot_results(df, "operand_corruption.png")

    verdict = {
        "residue_carried": bool(corrupt["top1_is_orig_W"].mean() >= 0.99 and corrupt["score_orig_WC"].mean() > 0),
        "score_collapses_on_corrupt": bool(delta.mean() < -1.0),
        "mean_delta_corrupt": float(delta.mean()),
        "recomputes_visible_pair": bool(corrupt["score_visible_W2C2"].mean() > corrupt["score_orig_WC"].mean()),
    }
    with open(OUT_DIR / "e6_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

"""E5 — Path ablation: does routing from t* to source regions matter?"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from E4_content_patching import (  # noqa: E402
    ITEMS,
    LANDING_LAYERS,
    PRIORITY_LAYERS,
    get_positions,
    run_baseline,
    score_at_tstar,
    tag_regions,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E5_outputs"
except NameError:
    OUT_DIR = Path("E5_outputs")
OUT_DIR.mkdir(exist_ok=True)

PRIORITY_HEADS = [(9, 9), (11, 0), (10, 7)]


@dataclass
class AblationResult:
    item: str
    intervention: str
    baseline: float
    ablated: float
    delta: float


def make_path_ablate_hooks(
    layers: list[int],
    dest: int,
    src_positions: list[int],
    heads: list[int] | None = None,
) -> list[tuple[str, object]]:
    """Zero attention t* → src_positions, renormalize remaining keys per head."""

    def hook_fn(activation, hook, d=dest, src=src_positions, head_ids=heads):
        # [batch, head, q, k]
        pat = activation
        if head_ids is None:
            sl = pat[0, :, d, :]
            sl[:, src] = 0.0
            sl = sl / sl.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pat[0, :, d, :] = sl
        else:
            for h in head_ids:
                row = pat[0, h, d, :]
                row[src] = 0.0
                pat[0, h, d, :] = row / row.sum().clamp(min=1e-8)
        return pat

    return [(f"blocks.{L}.attn.hook_pattern", hook_fn) for L in layers]


def run_ablated(model, tokens, hooks, t_star: int, W_id: int, C_id: int) -> float:
    with model.hooks(hooks):
        logits = model(tokens)
    return score_at_tstar(model, logits, t_star, W_id, C_id)


def random_control_positions(str_toks: list[str], t_star: int, width: int) -> list[int]:
    center = max(1, t_star - 4)
    pos = list(range(center, min(len(str_toks), center + width)))
    return [p for p in pos if p != t_star]


def run_item(model, item) -> list[AblationResult]:
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)
    tokens, str_toks, W_pos, t_star, regions = get_positions(model, item, "W")
    baseline = run_baseline(model, tokens, t_star, W_id, C_id)

    W_window = regions["W_window"]
    W_only = regions["W"]
    revision = regions["revision"]
    ops2 = regions["ops2"]
    rand_pos = random_control_positions(str_toks, t_star, len(W_window))

    results: list[AblationResult] = []

    def record(name: str, ablated: float):
        results.append(AblationResult(item.name, name, baseline, ablated, ablated - baseline))

    specs = [
        ("path_block_Wwin_L5-11", LANDING_LAYERS, W_window, None),
        ("path_block_Wwin_L9-11", PRIORITY_LAYERS, W_window, None),
        ("path_block_W_only_L9-11", PRIORITY_LAYERS, W_only, None),
        ("path_block_ops2_L5-11", LANDING_LAYERS, ops2, None),
        ("path_block_ops2_L9-11", PRIORITY_LAYERS, ops2, None),
        ("path_block_revision_L9-11", PRIORITY_LAYERS, revision, None),
        ("path_block_randpos_L9-11", PRIORITY_LAYERS, rand_pos, None),
    ]

    for name, layers, src, heads in specs:
        hooks = make_path_ablate_hooks(layers, t_star, src, heads)
        record(name, run_ablated(model, tokens, hooks, t_star, W_id, C_id))

    # Priority heads only → W_window, late layers
    for L, h in PRIORITY_HEADS:
        hooks = make_path_ablate_hooks([L], t_star, W_window, heads=[h])
        record(f"path_block_Wwin_L{L}H{h}", run_ablated(model, tokens, hooks, t_star, W_id, C_id))

    # Per-layer sweep → W_window
    for L in range(model.cfg.n_layers):
        hooks = make_path_ablate_hooks([L], t_star, W_window, None)
        record(f"path_block_Wwin_L{L}", run_ablated(model, tokens, hooks, t_star, W_id, C_id))

    return results


def plot_layer_sweep(df: pd.DataFrame, fname: str):
    sweep = df[df["intervention"].str.match(r"path_block_Wwin_L\d+$")]
    layers = sorted(int(n.split("L")[1]) for n in sweep["intervention"].unique())
    mean_d, sem_d = [], []
    for L in layers:
        sub = sweep[sweep["intervention"] == f"path_block_Wwin_L{L}"]["delta"]
        mean_d.append(sub.mean())
        sem_d.append(sub.std() / np.sqrt(len(sub)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layers, mean_d, yerr=sem_d, capsize=3, color="indianred", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("Δ score at t* (ablated − baseline)")
    ax.set_title("E5: per-layer path block t* → W_window")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_main_bars(summary: pd.DataFrame, fname: str):
    keep = [
        "path_block_Wwin_L5-11",
        "path_block_Wwin_L9-11",
        "path_block_W_only_L9-11",
        "path_block_ops2_L9-11",
        "path_block_revision_L9-11",
        "path_block_randpos_L9-11",
        "path_block_Wwin_L9H9",
        "path_block_Wwin_L11H0",
        "path_block_Wwin_L10H7",
    ]
    sub = summary[summary["intervention"].isin(keep)]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_delta"], yerr=sub["sem_delta"], capsize=3, color="indianred", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sub["intervention"]], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("mean Δ score at t*")
    ax.set_title("E5 path ablations (8-item mean)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE, fold_ln=False)
    model.eval()

    all_results: list[AblationResult] = []
    for item in ITEMS:
        print("item:", item.name)
        all_results.extend(run_item(model, item))

    rows = [
        {"item": r.item, "intervention": r.intervention, "baseline": r.baseline,
         "ablated": r.ablated, "delta": r.delta}
        for r in all_results
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "all_ablations.csv", index=False)

    summary = (
        df.groupby("intervention")
        .agg(mean_baseline=("baseline", "mean"), mean_ablated=("ablated", "mean"),
             mean_delta=("delta", "mean"), sem_delta=("delta", "sem"), n=("delta", "count"))
        .reset_index()
        .sort_values("mean_delta")
    )
    summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    key = summary[summary["intervention"].str.contains("L5-11|L9-11|L9H9|L11H0|rand|revision|ops2")]
    print(key.to_string(index=False))

    plot_layer_sweep(df, "per_layer_path_block.png")
    plot_main_bars(summary, "main_ablations.png")

    w511 = summary.loc[summary["intervention"] == "path_block_Wwin_L5-11", "mean_delta"].iloc[0]
    w911 = summary.loc[summary["intervention"] == "path_block_Wwin_L9-11", "mean_delta"].iloc[0]

    verdict = {
        "path_Wwin_L5-11_delta": float(w511),
        "path_Wwin_L9-11_delta": float(w911),
        "routing_necessary": bool(w911 < -0.5),
        "e4_content_vs_e5_routing": (
            "Compare E4 necessity_resid_Wwin_L5-11_Cswap (~-4.9) vs path_block_Wwin_L9-11 here"
        ),
    }
    with open(OUT_DIR / "e5_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

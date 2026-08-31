"""Forced W vs forced C anchoring asymmetry (post-spine extension)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl

from E4_content_patching import (
    ITEMS,
    LANDING_LAYERS,
    build_C_prompt,
    build_W_prompt,
    find_impulse_pos,
    find_t_star,
    get_positions,
    make_resid_patch_hooks,
    run_baseline,
    run_with_hooks,
    score_at_tstar,
    tag_regions,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "forced_W_vs_C_outputs"
except NameError:
    OUT_DIR = Path("forced_W_vs_C_outputs")
OUT_DIR.mkdir(exist_ok=True)


def evaluate_behavior(model, item, variant: str) -> dict:
    prompt = build_W_prompt(item) if variant == "W" else build_C_prompt(item)
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)
    impulse_pos = find_impulse_pos(str_toks)
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)
    logits = model(model.to_tokens(prompt))
    score = score_at_tstar(model, logits, t_star, W_id, C_id)
    top1 = model.to_string([int(logits[0, t_star].argmax())])
    impulse_tok = str_toks[impulse_pos]
    return {
        "item": item.name,
        "variant": variant,
        "prompt": prompt,
        "t_star": t_star,
        "impulse_pos": impulse_pos,
        "impulse_token": impulse_tok,
        "score_W_minus_C": score,
        "abs_score": abs(score),
        "score_C_minus_W": -score,
        "top1": top1,
        "top1_is_bank_W": top1 == item.W_str,
        "top1_is_bank_C": top1 == item.C_str,
        "top1_matches_impulse": top1 == impulse_tok,
    }


def run_causal_pair(model, item) -> list[dict]:
    """Mirrored residual swaps: W-prompt C-swap vs C-prompt W-swap."""
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)

    w_tokens, w_toks, W_pos, t_star_w, regions_w = get_positions(model, item, "W")
    c_tokens, c_toks, C_pos, t_star_c, regions_c = get_positions(model, item, "C")

    _, w_cache = model.run_with_cache(w_tokens)
    _, c_cache = model.run_with_cache(c_tokens)

    w_baseline = run_baseline(model, w_tokens, t_star_w, W_id, C_id)
    c_baseline = run_baseline(model, c_tokens, t_star_c, W_id, C_id)

    w_window = regions_w["W_window"]
    c_window = regions_c["W_window"]

    hooks_w = make_resid_patch_hooks(c_cache, LANDING_LAYERS, w_window)
    w_patched = run_with_hooks(model, w_tokens, hooks_w, t_star_w, W_id, C_id)

    hooks_c = make_resid_patch_hooks(w_cache, LANDING_LAYERS, c_window)
    c_patched = run_with_hooks(model, c_tokens, hooks_c, t_star_c, W_id, C_id)

    return [
        {
            "item": item.name,
            "prompt_variant": "forced_W",
            "intervention": "impulse_window_Cswap",
            "baseline": w_baseline,
            "patched": w_patched,
            "delta": w_patched - w_baseline,
            "abs_delta": abs(w_patched - w_baseline),
        },
        {
            "item": item.name,
            "prompt_variant": "forced_C",
            "intervention": "impulse_window_Wswap",
            "baseline": c_baseline,
            "patched": c_patched,
            "delta": c_patched - c_baseline,
            "abs_delta": abs(c_patched - c_baseline),
        },
    ]


def plot_behavior(behavior: pd.DataFrame):
    summary = behavior.groupby("variant").agg(
        mean_score=("score_W_minus_C", "mean"),
        sem_score=("score_W_minus_C", "sem"),
        mean_abs=("abs_score", "mean"),
        frac_top1_impulse=("top1_matches_impulse", "mean"),
    )
    colors = {"W": "#e74c3c", "C": "#2ecc71"}
    labels = {"W": "forced wrong (W)", "C": "forced correct (C)"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    variants = ["W", "C"]
    x = np.arange(len(variants))
    means = [summary.loc[v, "mean_score"] for v in variants]
    sems = [summary.loc[v, "sem_score"] for v in variants]
    axes[0].bar(x, means, yerr=sems, capsize=4, color=[colors[v] for v in variants], alpha=0.85)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[v] for v in variants])
    axes[0].set_ylabel("score at t* (logit W − logit C)")
    axes[0].set_title("Behavioral anchoring")

    frac = [summary.loc[v, "frac_top1_impulse"] for v in variants]
    axes[1].bar(x, frac, color=[colors[v] for v in variants], alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[v] for v in variants])
    axes[1].set_ylabel("frac top-1 = impulse token")
    axes[1].set_title("Persistence at t*")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "forced_W_vs_C_behavior.png", dpi=150)
    plt.close(fig)

    # per-item paired
    w = behavior[behavior["variant"] == "W"].set_index("item")
    c = behavior[behavior["variant"] == "C"].set_index("item")
    items = w.index.tolist()
    xi = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xi - 0.2, [w.loc[i, "score_W_minus_C"] for i in items], width=0.4, label="forced W", color=colors["W"])
    ax.bar(xi + 0.2, [c.loc[i, "score_W_minus_C"] for i in items], width=0.4, label="forced C", color=colors["C"])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels(items, rotation=30, ha="right")
    ax.set_ylabel("logit W − logit C @ t*")
    ax.legend()
    ax.set_title("Forced W vs C per-item paired scores")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forced_W_vs_C_per_item.png", dpi=150)
    plt.close(fig)


def plot_causal(causal: pd.DataFrame):
    summary = causal.groupby("prompt_variant").agg(
        mean_delta=("delta", "mean"),
        sem_delta=("delta", "sem"),
        mean_abs_delta=("abs_delta", "mean"),
    )
    labels = {"forced_W": "W-prompt\n(C-swap)", "forced_C": "C-prompt\n(W-swap)"}
    order = ["forced_W", "forced_C"]
    colors = {"forced_W": "#e74c3c", "forced_C": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    means = [summary.loc[k, "mean_delta"] for k in order]
    sems = [summary.loc[k, "sem_delta"] for k in order]
    ax.bar(x, means, yerr=sems, capsize=4, color=[colors[k] for k in order], alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order])
    ax.set_ylabel("Δ score at t* (patched − baseline)")
    ax.set_title("Mirrored causal flip (L5–L11 impulse-window resid swap)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "forced_W_vs_C_causal.png", dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE, fold_ln=False)
    model.eval()

    behavior_rows = []
    causal_rows = []
    for item in ITEMS:
        print("item:", item.name)
        for variant in ("W", "C"):
            behavior_rows.append(evaluate_behavior(model, item, variant))
        causal_rows.extend(run_causal_pair(model, item))

    behavior = pd.DataFrame(behavior_rows)
    causal = pd.DataFrame(causal_rows)
    behavior.to_csv(OUT_DIR / "behavior.csv", index=False)
    causal.to_csv(OUT_DIR / "causal.csv", index=False)

    w = behavior[behavior["variant"] == "W"].set_index("item")
    c = behavior[behavior["variant"] == "C"].set_index("item")

    mean_w = float(w["score_W_minus_C"].mean())
    mean_c = float(c["score_W_minus_C"].mean())
    abs_w = float(w["abs_score"].mean())
    abs_c = float(c["abs_score"].mean())
    asym_ratio = abs_w / abs_c if abs_c > 1e-6 else float("nan")

    cw = causal[causal["prompt_variant"] == "forced_W"]
    cc = causal[causal["prompt_variant"] == "forced_C"]

    summary = pd.DataFrame(
        {
            "mean_score_forced_W": [mean_w],
            "mean_score_forced_C": [mean_c],
            "mean_abs_score_forced_W": [abs_w],
            "mean_abs_score_forced_C": [abs_c],
            "asymmetry_ratio_abs_W_over_C": [asym_ratio],
            "frac_top1_impulse_forced_W": [w["top1_matches_impulse"].mean()],
            "frac_top1_impulse_forced_C": [c["top1_matches_impulse"].mean()],
            "mean_causal_delta_W_prompt_Cswap": [cw["delta"].mean()],
            "mean_causal_delta_C_prompt_Wswap": [cc["delta"].mean()],
            "mean_abs_causal_delta_W": [cw["abs_delta"].mean()],
            "mean_abs_causal_delta_C": [cc["abs_delta"].mean()],
        }
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.T)

    per_item = pd.DataFrame(
        {
            "score_forced_W": w["score_W_minus_C"],
            "score_forced_C": c["score_W_minus_C"],
            "top1_forced_W": w["top1"],
            "top1_forced_C": c["top1"],
            "delta_W_Cswap": cw.set_index("item")["delta"],
            "delta_C_Wswap": cc.set_index("item")["delta"],
        }
    )
    per_item.to_csv(OUT_DIR / "per_item.csv")

    plot_behavior(behavior)
    plot_causal(causal)

    symmetric_behavior = abs(abs_w - abs_c) < 0.5
    symmetric_causal = abs(cw["abs_delta"].mean() - cc["abs_delta"].mean()) < 1.0
    verdict = {
        "forced_W_persists": bool(w["top1_matches_impulse"].mean() >= 0.99 and mean_w > 1.0),
        "forced_C_persists": bool(c["top1_matches_impulse"].mean() >= 0.99 and mean_c < -1.0),
        "behavior_near_symmetric": bool(symmetric_behavior),
        "causal_near_symmetric": bool(symmetric_causal),
        "mean_score_forced_W": mean_w,
        "mean_score_forced_C": mean_c,
        "asymmetry_ratio_abs_W_over_C": asym_ratio,
        "frac_top1_impulse_forced_W": float(w["top1_matches_impulse"].mean()),
        "frac_top1_impulse_forced_C": float(c["top1_matches_impulse"].mean()),
        "mean_causal_delta_W_prompt_Cswap": float(cw["delta"].mean()),
        "mean_causal_delta_C_prompt_Wswap": float(cc["delta"].mean()),
        "interpretation": (
            "near_symmetric_wrong_right_anchoring"
            if symmetric_behavior and symmetric_causal
            else "asymmetric_wrong_vs_right_anchoring"
        ),
    }
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

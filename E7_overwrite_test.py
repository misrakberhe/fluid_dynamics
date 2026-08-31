"""E7 — Overwrite test: does the revision cue write toward C?"""

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
    CUE,
    ITEMS,
    find_impulse_pos,
    find_t_star,
    tag_regions,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E7_outputs"
except NameError:
    OUT_DIR = Path("E7_outputs")
OUT_DIR.mkdir(exist_ok=True)

FILLER_UNIT = " ..."
N_LAYERS = 12


def n_tokens(model, text: str) -> int:
    return len(model.to_str_tokens(text)) - 1


def build_prompt(model, item, with_cue: bool = True) -> str:
    prefix = f"{item.a} + {item.b} ={item.W_str}."
    suffix = f" {item.a} + {item.b} ="
    middle = CUE if with_cue else FILLER_UNIT * n_tokens(model, CUE)
    return prefix + middle + suffix


def w_direction(model, W_str: str, C_str: str) -> torch.Tensor:
    W_id = model.to_single_token(W_str)
    C_id = model.to_single_token(C_str)
    return model.W_U[:, W_id] - model.W_U[:, C_id]


def region_sum(attn_raw: np.ndarray, mlp_raw: np.ndarray, positions: list[int]) -> float:
    if not positions:
        return 0.0
    post = attn_raw[:, positions] + mlp_raw[:, positions]
    return float(post.sum())


def region_layer_profile(attn_raw: np.ndarray, mlp_raw: np.ndarray, positions: list[int]) -> np.ndarray:
    if not positions:
        return np.zeros(N_LAYERS)
    post = attn_raw[:, positions] + mlp_raw[:, positions]
    return post.sum(axis=1)


def region_unit_mean(attn_unit: np.ndarray, mlp_unit: np.ndarray, positions: list[int]) -> float:
    if not positions:
        return 0.0
    post = attn_unit[:, positions] + mlp_unit[:, positions]
    return float(post.mean())


@dataclass
class DLARecord:
    item: str
    with_cue: bool
    prompt: str
    W_pos: int
    t_star: int
    final_score: float
    revision_positions: list[int]
    middle_positions: list[int]
    sum_post_revision: float
    sum_post_W: float
    sum_post_W_window: float
    sum_post_ops2: float
    unit_mean_revision: float
    layer_profile_revision: np.ndarray
    n_revision_tokens: int


def run_dla(model, item, with_cue: bool = True) -> DLARecord:
    prompt = build_prompt(model, item, with_cue=with_cue)
    tokens = model.to_tokens(prompt)
    str_toks = model.to_str_tokens(prompt)

    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)
    w = w_direction(model, item.W_str, item.C_str)
    w_hat = w / w.norm()

    W_pos = find_impulse_pos(str_toks)
    t_star = find_t_star(str_toks)
    ops2_pos = list(range(t_star - 3, t_star))
    W_window = list(range(max(0, W_pos - 2), min(len(str_toks), W_pos + 3)))

    if with_cue:
        regions = tag_regions(str_toks, W_pos, t_star)
        middle_positions = regions["revision"]
    else:
        dot_pos = next(j for j, t in enumerate(str_toks) if t == ".")
        middle_positions = list(range(dot_pos + 1, ops2_pos[0]))
        regions = {
            "W": [W_pos],
            "W_window": W_window,
            "revision": middle_positions,
            "ops2": ops2_pos,
        }

    logits, cache = model.run_with_cache(tokens)
    final_score = float(logits[0, t_star, W_id] - logits[0, t_star, C_id])

    seq = cache["hook_embed"].shape[1]
    attn_raw = np.zeros((N_LAYERS, seq), dtype=np.float64)
    mlp_raw = np.zeros((N_LAYERS, seq), dtype=np.float64)
    attn_unit = np.zeros((N_LAYERS, seq), dtype=np.float64)
    mlp_unit = np.zeros((N_LAYERS, seq), dtype=np.float64)

    for L in range(N_LAYERS):
        a = cache[f"blocks.{L}.hook_attn_out"][0]
        m = cache[f"blocks.{L}.hook_mlp_out"][0]
        attn_raw[L] = (a @ w).detach().float().cpu().numpy()
        mlp_raw[L] = (m @ w).detach().float().cpu().numpy()
        a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        m_n = m / (m.norm(dim=-1, keepdim=True) + 1e-8)
        attn_unit[L] = (a_n @ w_hat).detach().float().cpu().numpy()
        mlp_unit[L] = (m_n @ w_hat).detach().float().cpu().numpy()

    revision_positions = middle_positions

    return DLARecord(
        item=item.name,
        with_cue=with_cue,
        prompt=prompt,
        W_pos=W_pos,
        t_star=t_star,
        final_score=final_score,
        revision_positions=revision_positions,
        middle_positions=middle_positions,
        sum_post_revision=region_sum(attn_raw, mlp_raw, middle_positions),
        sum_post_W=region_sum(attn_raw, mlp_raw, regions["W"]),
        sum_post_W_window=region_sum(attn_raw, mlp_raw, regions["W_window"]),
        sum_post_ops2=region_sum(attn_raw, mlp_raw, regions["ops2"]),
        unit_mean_revision=region_unit_mean(attn_unit, mlp_unit, middle_positions),
        layer_profile_revision=region_layer_profile(attn_raw, mlp_raw, middle_positions),
        n_revision_tokens=len(middle_positions),
    )


def plot_layer_profiles(cue_df: pd.DataFrame, fname: str):
    profiles = np.stack(cue_df["layer_profile_revision"].values)
    mean_p = profiles.mean(axis=0)
    sem_p = profiles.std(axis=0) / np.sqrt(len(profiles))

    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(N_LAYERS)
    ax.fill_between(layers, mean_p - sem_p, mean_p + sem_p, alpha=0.25, color="steelblue")
    ax.plot(layers, mean_p, "o-", color="steelblue", label="revision span (cue)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("sum (attn+mlp)·w over revision tokens")
    ax.set_title("E7 — per-layer writes at revision span (positive → W, negative → C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_region_comparison(cue_df: pd.DataFrame, fname: str):
    regions = ["revision", "W", "W_window", "ops2"]
    cols = [f"sum_post_{r}" if r != "revision" else "sum_post_revision" for r in regions]
    means = [cue_df[c].mean() for c in cols]
    sems = [cue_df[c].sem() for c in cols]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(regions))
    ax.bar(x, means, yerr=sems, capsize=4, color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"], alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("mean Σ (attn+mlp)·w")
    ax.set_title("E7 — total post-embed write by region (cue prompt)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_cue_vs_filler(cue_df: pd.DataFrame, filler_df: pd.DataFrame, fname: str):
    merged = cue_df.merge(filler_df, on="item", suffixes=("_cue", "_filler"))
    delta = merged["sum_post_revision_cue"] - merged["sum_post_revision_filler"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(np.arange(len(merged)), merged["sum_post_revision_cue"], alpha=0.7, label="cue")
    axes[0].bar(
        np.arange(len(merged)),
        merged["sum_post_revision_filler"],
        alpha=0.7,
        label="filler (no cue)",
    )
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Σ (attn+mlp)·w at middle span")
    axes[0].set_title("Per-item middle-span writes")
    axes[0].legend()

    axes[1].bar(np.arange(len(merged)), delta, color="teal", alpha=0.85)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("cue − filler")
    axes[1].set_title(f"mean Δ = {delta.mean():+.2f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    records = []
    for item in ITEMS:
        for with_cue in (True, False):
            print(item.name, "cue" if with_cue else "filler")
            rec = run_dla(model, item, with_cue=with_cue)
            records.append({
                "item": rec.item,
                "with_cue": rec.with_cue,
                "final_score": rec.final_score,
                "n_middle_tokens": rec.n_revision_tokens,
                "sum_post_revision": rec.sum_post_revision,
                "sum_post_W": rec.sum_post_W,
                "sum_post_W_window": rec.sum_post_W_window,
                "sum_post_ops2": rec.sum_post_ops2,
                "unit_mean_revision": rec.unit_mean_revision,
                "layer_profile_revision": rec.layer_profile_revision,
                "toward_C": rec.sum_post_revision < 0,
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "overwrite_dla.csv", index=False)

    cue = df[df["with_cue"]].copy()
    filler = df[~df["with_cue"]].copy()

    cue_delta = cue.set_index("item")["sum_post_revision"] - filler.set_index("item")["sum_post_revision"]

    summary = pd.DataFrame({
        "mean_sum_revision_cue": [cue["sum_post_revision"].mean()],
        "sem_sum_revision_cue": [cue["sum_post_revision"].sem()],
        "frac_revision_toward_C_cue": [cue["toward_C"].mean()],
        "mean_unit_revision_cue": [cue["unit_mean_revision"].mean()],
        "mean_sum_revision_filler": [filler["sum_post_revision"].mean()],
        "mean_cue_minus_filler": [cue_delta.mean()],
        "sem_cue_minus_filler": [cue_delta.std() / np.sqrt(len(cue_delta))],
        "mean_sum_W": [cue["sum_post_W"].mean()],
        "mean_sum_W_window": [cue["sum_post_W_window"].mean()],
        "mean_sum_ops2": [cue["sum_post_ops2"].mean()],
        "mean_final_score": [cue["final_score"].mean()],
    })
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.T)

    per_item = cue.set_index("item")[
        ["sum_post_revision", "sum_post_W", "sum_post_W_window", "unit_mean_revision", "toward_C", "final_score"]
    ]
    per_item["cue_minus_filler"] = cue_delta
    per_item.to_csv(OUT_DIR / "per_item.csv")

    plot_layer_profiles(cue, "revision_layer_profile.png")
    plot_region_comparison(cue, "region_comparison.png")
    plot_cue_vs_filler(cue, filler, "cue_vs_filler.png")

    # Verdict: competing push toward C if net revision write is meaningfully negative
    mean_rev = float(cue["sum_post_revision"].mean())
    frac_neg = float(cue["toward_C"].mean())
    competing = mean_rev < -5.0 or frac_neg >= 0.75
    verdict = {
        "competing_push_toward_C": competing,
        "mean_sum_post_revision_cue": mean_rev,
        "frac_items_revision_toward_C": frac_neg,
        "mean_unit_direction_revision": float(cue["unit_mean_revision"].mean()),
        "mean_cue_minus_filler_revision": float(cue_delta.mean()),
        "revision_weaker_than_W_window": bool(
            abs(cue["sum_post_revision"].mean()) < abs(cue["sum_post_W_window"].mean())
        ),
        "interpretation": (
            "cue_recruits_competing_C_wave"
            if competing
            else "cue_not_recruited_no_competing_overwrite"
        ),
    }
    with open(OUT_DIR / "e7_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

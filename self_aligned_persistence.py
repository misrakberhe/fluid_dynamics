"""Self-aligned persistence — score gen vs contrast on each item's own axis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl

from E4_content_patching import (
    CUE,
    ITEMS,
    LANDING_LAYERS,
    build_W_prompt,
    find_impulse_pos,
    find_t_star,
    make_resid_patch_hooks,
    run_with_hooks,
    tag_regions,
)
from E8_self_generated_impulse import GenResult, build_prefix, greedy_generate_impulse

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "self_aligned_persistence_outputs"
except NameError:
    OUT_DIR = Path("self_aligned_persistence_outputs")
OUT_DIR.mkdir(exist_ok=True)


def build_gen_prompt(item, gen: GenResult) -> str:
    suffix = f"{CUE} {item.a} + {item.b} ="
    return f"{gen.prefix}{gen.impulse_str}{suffix}"


def build_contrast_prompt(item, contrast_str: str) -> str:
    return f"{item.a} + {item.b} ={contrast_str}.{CUE} {item.a} + {item.b} ="


def aligned_score(logits, t_star: int, pos_id: int, contrast_id: int) -> float:
    v = logits[0, t_star]
    return float(v[pos_id] - v[contrast_id])


def runner_up_at_gen(model, prefix: str, gen_token: str) -> str:
    logits = model(model.to_tokens(prefix))
    v = logits[0, -1]
    order = v.argsort(descending=True)
    gen_id = model.to_single_token(gen_token)
    for idx in order:
        tok = model.to_string([int(idx)])
        if int(idx) != gen_id:
            return tok
    return model.to_string([int(order[1])])


def evaluate_behavior(
    model,
    item,
    gen: GenResult,
    mode: str,
    prompt: str,
    pos_id: int,
    contrast_id: int,
    contrast_str: str,
) -> dict:
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)
    impulse_pos = find_impulse_pos(str_toks)
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)
    logits = model(model.to_tokens(prompt))
    score_aligned = aligned_score(logits, t_star, pos_id, contrast_id)
    score_bank = float(logits[0, t_star, W_id] - logits[0, t_star, C_id])
    top1 = model.to_string([int(logits[0, t_star].argmax())])
    return {
        "item": item.name,
        "mode": mode,
        "gen_first_token": gen.first_token,
        "contrast_token": contrast_str,
        "score_aligned": score_aligned,
        "score_bank_WC": score_bank,
        "top1": top1,
        "top1_is_gen": top1 == gen.first_token,
        "top1_is_bank_W": top1 == item.W_str,
        "top1_is_contrast": top1 == contrast_str,
        "impulse_pos": impulse_pos,
        "t_star": t_star,
    }


def run_causal_swap(model, item, gen: GenResult, contrast_str: str) -> dict:
    gen_prompt = build_gen_prompt(item, gen)
    contrast_prompt = build_contrast_prompt(item, contrast_str)

    gen_tokens = model.to_tokens(gen_prompt)
    contrast_tokens = model.to_tokens(contrast_prompt)
    gen_toks = model.to_str_tokens(gen_prompt)
    contrast_toks = model.to_str_tokens(contrast_prompt)

    gen_pos = find_impulse_pos(gen_toks)
    t_star = find_t_star(gen_toks)
    gen_window = tag_regions(gen_toks, gen_pos, t_star)["W_window"]

    pos_id = model.to_single_token(gen.first_token)
    contrast_id = model.to_single_token(contrast_str)

    _, contrast_cache = model.run_with_cache(contrast_tokens)
    gen_logits = model(gen_tokens)
    baseline = aligned_score(gen_logits, t_star, pos_id, contrast_id)

    hooks = make_resid_patch_hooks(contrast_cache, LANDING_LAYERS, gen_window)
    with model.hooks(hooks):
        patched_logits = model(gen_tokens)
    patched = aligned_score(patched_logits, t_star, pos_id, contrast_id)

    return {
        "item": item.name,
        "baseline_aligned": baseline,
        "patched_aligned": patched,
        "delta_aligned": patched - baseline,
        "abs_delta": abs(patched - baseline),
        "gen_first_token": gen.first_token,
        "contrast_token": contrast_str,
    }


def plot_behavior(behavior: pd.DataFrame):
    modes = ["forced_bank_W", "forced_gen", "self_generated"]
    colors = {"forced_bank_W": "#3498db", "forced_gen": "#95a5a6", "self_generated": "#e74c3c"}
    summary = behavior.groupby("mode").agg(
        mean_aligned=("score_aligned", "mean"),
        sem_aligned=("score_aligned", "sem"),
        mean_bank=("score_bank_WC", "mean"),
        sem_bank=("score_bank_WC", "sem"),
        frac_top1_gen=("top1_is_gen", "mean"),
    ).reindex(modes)

    display_scores = [
        summary.loc["forced_bank_W", "mean_bank"],
        summary.loc["forced_gen", "mean_aligned"],
        summary.loc["self_generated", "mean_aligned"],
    ]
    display_sems = [
        summary.loc["forced_bank_W", "sem_bank"],
        summary.loc["forced_gen", "sem_aligned"],
        summary.loc["self_generated", "sem_aligned"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(modes))
    axes[0].bar(
        x,
        display_scores,
        yerr=display_sems,
        capsize=4,
        color=[colors[m] for m in modes],
        alpha=0.9,
    )
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["forced bank W\n(bank C contrast)", "forced gen\n(aligned)", "self gen\n(aligned)"], fontsize=8)
    axes[0].set_ylabel("score @ t* (each on own axis)")
    axes[0].set_title("Self-aligned persistence")

    axes[1].bar(x, summary["frac_top1_gen"], color=[colors[m] for m in modes], alpha=0.9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("frac top-1 = gen token")
    axes[1].set_title("Stickiness on aligned axis")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "self_aligned_behavior.png", dpi=150)
    plt.close(fig)

    forced_gen = behavior[behavior["mode"] == "forced_gen"].set_index("item")
    self_gen = behavior[behavior["mode"] == "self_generated"].set_index("item")
    bank_w = behavior[behavior["mode"] == "forced_bank_W"].set_index("item")
    items = forced_gen.index.tolist()
    xi = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xi - 0.25, [bank_w.loc[i, "score_bank_WC"] for i in items], width=0.25, label="forced bank W (W−C)", color=colors["forced_bank_W"])
    ax.bar(xi, [forced_gen.loc[i, "score_aligned"] for i in items], width=0.25, label="forced gen (aligned)", color=colors["forced_gen"])
    ax.bar(xi + 0.25, [self_gen.loc[i, "score_aligned"] for i in items], width=0.25, label="self gen (aligned)", color=colors["self_generated"])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels(items, rotation=30, ha="right")
    ax.set_ylabel("score @ t* (each on own axis)")
    ax.legend(fontsize=8)
    ax.set_title("Per-item aligned scores")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "self_aligned_per_item.png", dpi=150)
    plt.close(fig)


def plot_causal(causal: pd.DataFrame, mean_forced_gen: float, mean_bank_w_bank: float, reference_delta: float):
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_d = causal["delta_aligned"].mean()
    ax.bar(
        ["gen causal\n|Δ|", "forced gen\n(aligned)", "forced bank W\n(W−C)"],
        [abs(mean_d), mean_forced_gen, mean_bank_w_bank],
        color=["#c0392b", "#e74c3c", "#3498db"],
        alpha=0.85,
    )
    ax.axhline(abs(reference_delta), color="#95a5a6", ls="--", lw=1.2, label=f"bank W causal |Δ|={abs(reference_delta):.1f}")
    ax.set_ylabel("score / |Δ| @ t*")
    ax.set_title("Self-aligned vs bank-W strength")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "self_aligned_causal.png", dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE, fold_ln=False)
    model.eval()

    behavior_rows = []
    causal_rows = []

    for item in ITEMS:
        prefix = build_prefix(item)
        gen = greedy_generate_impulse(model, prefix)
        contrast_str = item.C_str
        contrast_runner = runner_up_at_gen(model, prefix, gen.first_token)

        pos_id = model.to_single_token(gen.first_token)
        contrast_id = model.to_single_token(contrast_str)

        gen_prompt = build_gen_prompt(item, gen)
        self_prompt = f"{gen.full_prefix}{CUE} {item.a} + {item.b} ="
        bank_w_prompt = build_W_prompt(item)

        print(item.name, "gen:", repr(gen.first_token), "contrast:", repr(contrast_str))

        for mode, prompt in [
            ("forced_bank_W", bank_w_prompt),
            ("forced_gen", gen_prompt),
            ("self_generated", self_prompt),
        ]:
            row = evaluate_behavior(model, item, gen, mode, prompt, pos_id, contrast_id, contrast_str)
            row["contrast_runner_up"] = contrast_runner
            behavior_rows.append(row)

        causal_rows.append(run_causal_swap(model, item, gen, contrast_str))

    behavior = pd.DataFrame(behavior_rows)
    causal = pd.DataFrame(causal_rows)
    behavior.to_csv(OUT_DIR / "behavior.csv", index=False)
    causal.to_csv(OUT_DIR / "causal.csv", index=False)

    forced_gen = behavior[behavior["mode"] == "forced_gen"].set_index("item")
    self_gen = behavior[behavior["mode"] == "self_generated"].set_index("item")
    bank_w = behavior[behavior["mode"] == "forced_bank_W"].set_index("item")

    mean_forced_gen = float(forced_gen["score_aligned"].mean())
    mean_self = float(self_gen["score_aligned"].mean())
    mean_bank_w_aligned = float(bank_w["score_aligned"].mean())
    mean_bank_w_bank = float(bank_w["score_bank_WC"].mean())
    mean_causal = float(causal["delta_aligned"].mean())
    ref_w_causal = -4.937479  # from forced_W_vs_C_outputs

    summary = pd.DataFrame(
        {
            "mean_aligned_forced_gen": [mean_forced_gen],
            "mean_aligned_self_gen": [mean_self],
            "mean_aligned_forced_bank_W": [mean_bank_w_aligned],
            "mean_bank_WC_forced_bank_W": [mean_bank_w_bank],
            "mean_delta_self_minus_forced_gen": [(self_gen["score_aligned"] - forced_gen["score_aligned"]).mean()],
            "frac_top1_gen_forced_gen": [forced_gen["top1_is_gen"].mean()],
            "frac_top1_gen_self_gen": [self_gen["top1_is_gen"].mean()],
            "frac_top1_gen_forced_bank_W": [bank_w["top1_is_gen"].mean()],
            "mean_causal_delta_gen_Cswap": [mean_causal],
            "mean_abs_causal_delta_gen": [causal["abs_delta"].mean()],
            "reference_causal_delta_bank_W_Cswap": [ref_w_causal],
        }
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.T)

    per_item = pd.DataFrame(
        {
            "gen_first_token": forced_gen["gen_first_token"],
            "contrast": forced_gen["contrast_token"],
            "aligned_forced_gen": forced_gen["score_aligned"],
            "aligned_self_gen": self_gen["score_aligned"],
            "aligned_forced_bank_W": bank_w["score_aligned"],
            "bank_WC_forced_bank_W": bank_w["score_bank_WC"],
            "causal_delta": causal.set_index("item")["delta_aligned"],
            "top1_forced_gen": forced_gen["top1"],
            "top1_self_gen": self_gen["top1"],
        }
    )
    per_item.to_csv(OUT_DIR / "per_item.csv")

    plot_behavior(behavior)
    plot_causal(causal, mean_forced_gen, mean_bank_w_bank, ref_w_causal)

    aligned_gap = mean_bank_w_bank - mean_forced_gen
    causal_ratio = abs(mean_causal) / abs(ref_w_causal) if ref_w_causal else float("nan")
    verdict = {
        "self_gen_matches_forced_gen_on_aligned_axis": bool(
            abs(mean_self - mean_forced_gen) < 0.15 and self_gen["top1_is_gen"].mean() >= 0.99
        ),
        "gen_persistence_at_least_as_strong_as_bank_W": bool(mean_forced_gen >= mean_bank_w_bank - 0.5),
        "gen_causally_localized": bool(causal["abs_delta"].mean() > 2.0),
        "causal_flip_stronger_than_bank_W": bool(abs(mean_causal) > abs(ref_w_causal) * 1.2),
        "mean_aligned_forced_gen": mean_forced_gen,
        "mean_aligned_self_gen": mean_self,
        "mean_bank_WC_forced_bank_W": mean_bank_w_bank,
        "strength_gap_bank_W_minus_gen_aligned": aligned_gap,
        "mean_causal_delta_gen_Cswap": mean_causal,
        "causal_ratio_vs_bank_W": causal_ratio,
        "frac_top1_gen_self": float(self_gen["top1_is_gen"].mean()),
        "interpretation": (
            "generic_slot_binding_on_aligned_axis"
            if self_gen["top1_is_gen"].mean() >= 0.99 and causal["abs_delta"].mean() > 2.0
            else "weak_or_asymmetric_self_aligned_persistence"
        ),
    }
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

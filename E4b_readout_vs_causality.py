"""E4b follow-up — readout (tuned-lens W−C at t*) vs causal necessity (patch Δ)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl
from tuned_lens.nn import TunedLens, Unembed, LogitLens

sys.path.insert(0, str(Path(__file__).resolve().parent))
from E4_content_patching import (  # noqa: E402
    ITEMS,
    get_positions,
    make_resid_patch_hooks,
    run_baseline,
    run_with_hooks,
)
from E3_persistence_routing import emergence_depth  # noqa: E402

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E4b_outputs"
except NameError:
    OUT_DIR = Path("E4b_outputs")
OUT_DIR.mkdir(exist_ok=True)

CAUSAL_THRESHOLD = 1.0  # |Δ score| for "nontrivial" single-layer patch


def w_direction(model, W_str: str, C_str: str) -> torch.Tensor:
    W_id = model.to_single_token(W_str)
    C_id = model.to_single_token(C_str)
    return model.W_U[:, W_id] - model.W_U[:, C_id]


def w_sensitivity_curve(
    model,
    w_tokens,
    t_star: int,
    W_id: int,
    C_id: int,
    w: torch.Tensor,
) -> np.ndarray:
    """Gradient ∂(W−C)/∂h · ŵ at t* per layer — local directional sensitivity proxy.

    Not full J-lens (no corpus-averaged J_l transport); cheap Jacobian-adjacent curve.
    See: https://github.com/anthropics/jacobian-lens
    """
    n_layers = model.cfg.n_layers
    w_hat = (w / w.norm()).detach()
    _, cache = model.run_with_cache(w_tokens)
    sens = np.zeros(n_layers, dtype=np.float64)

    for L in range(n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"][0, t_star].clone().detach().requires_grad_(True)

        def patch_hook(activation, hook, pos=t_star, r=resid):
            activation[0, pos, :] = r
            return activation

        with torch.enable_grad():
            with model.hooks([(f"blocks.{L}.hook_resid_post", patch_hook)]):
                logits = model(w_tokens)
            score = logits[0, t_star, W_id] - logits[0, t_star, C_id]
            score.backward()
        sens[L] = float(resid.grad @ w_hat)
    return sens


def tuned_lens_curve(model, tuned_lens, tokens, t_star: int, W_id: int, C_id: int, final_score: float) -> np.ndarray:
    n_layers = model.cfg.n_layers
    _, cache = model.run_with_cache(tokens)
    scores = np.zeros(n_layers + 1, dtype=np.float64)
    for L in range(n_layers):
        hidden = cache[f"blocks.{L}.hook_resid_pre"]
        tl_logits = tuned_lens.forward(hidden, L)
        scores[L] = float(tl_logits[0, t_star, W_id] - tl_logits[0, t_star, C_id])
    scores[-1] = final_score
    return scores


def logit_lens_curve(model, logit_lens, tokens, t_star: int, W_id: int, C_id: int, final_score: float) -> np.ndarray:
    n_layers = model.cfg.n_layers
    _, cache = model.run_with_cache(tokens)
    scores = np.zeros(n_layers + 1, dtype=np.float64)
    for L in range(n_layers):
        hidden = cache[f"blocks.{L}.hook_resid_pre"]
        ll_logits = logit_lens.forward(hidden, L)
        scores[L] = float(ll_logits[0, t_star, W_id] - ll_logits[0, t_star, C_id])
    scores[-1] = final_score
    return scores


def per_layer_patch_delta(model, w_tokens, c_cache, t_star: int, W_id: int, C_id: int) -> np.ndarray:
    baseline = run_baseline(model, w_tokens, t_star, W_id, C_id)
    n_layers = model.cfg.n_layers
    deltas = np.zeros(n_layers, dtype=np.float64)
    for L in range(n_layers):
        hooks = make_resid_patch_hooks(c_cache, [L], [t_star])
        patched = run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id)
        deltas[L] = patched - baseline
    return deltas


def first_layer_above(arr: np.ndarray, threshold: float, mode: str = "above") -> int | None:
    for i, v in enumerate(arr):
        if mode == "above" and v >= threshold:
            return i
        if mode == "below" and v <= -threshold:
            return i
    return None


def normalize01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE, fold_ln=False)
    model.eval()
    tuned_lens = TunedLens.from_unembed_and_pretrained(unembed=Unembed(model), lens_resource_id="gpt2").to(DEVICE)
    logit_lens = LogitLens.from_model(model).to(DEVICE)

    rows = []
    tuned_all = []
    logit_all = []
    patch_all = []
    sens_all = []

    for item in ITEMS:
        print("item:", item.name)
        W_id = model.to_single_token(item.W_str)
        C_id = model.to_single_token(item.C_str)
        w = w_direction(model, item.W_str, item.C_str)
        w_tokens, _, _, t_star, _ = get_positions(model, item, "W")
        c_tokens, _, _, _, _ = get_positions(model, item, "C")
        _, c_cache = model.run_with_cache(c_tokens)

        final = run_baseline(model, w_tokens, t_star, W_id, C_id)
        tuned = tuned_lens_curve(model, tuned_lens, w_tokens, t_star, W_id, C_id, final)
        ll = logit_lens_curve(model, logit_lens, w_tokens, t_star, W_id, C_id, final)
        patch_d = per_layer_patch_delta(model, w_tokens, c_cache, t_star, W_id, C_id)
        sens = w_sensitivity_curve(model, w_tokens, t_star, W_id, C_id, w)

        tuned_all.append(tuned)
        logit_all.append(ll)
        patch_all.append(patch_d)
        sens_all.append(sens)

        emerge_tuned = emergence_depth(tuned)
        emerge_logit = emergence_depth(ll)
        causal_onset = first_layer_above(patch_d, CAUSAL_THRESHOLD, mode="below")
        causal_peak = int(np.argmin(patch_d))

        rows.append({
            "item": item.name,
            "final_score": final,
            "emergence_tuned": emerge_tuned,
            "emergence_logit": emerge_logit,
            "causal_onset_layer": causal_onset,
            "causal_peak_layer": causal_peak,
            "patch_delta_L5": patch_d[5],
            "patch_delta_L9": patch_d[9],
            "tuned_L5": tuned[5],
            "tuned_L9": tuned[9],
        })

    df_items = pd.DataFrame(rows)
    df_items.to_csv(OUT_DIR / "readout_vs_causality_per_item.csv", index=False)

    n_layers = model.cfg.n_layers
    xs = np.arange(n_layers + 1)
    xs_patch = np.arange(n_layers)

    mean_tuned = np.mean(tuned_all, axis=0)
    mean_logit = np.mean(logit_all, axis=0)
    mean_patch = np.mean(patch_all, axis=0)
    mean_sens = np.mean(sens_all, axis=0)
    sem_patch = np.std(patch_all, axis=0) / np.sqrt(len(ITEMS))
    sem_sens = np.std(sens_all, axis=0) / np.sqrt(len(ITEMS))

    # --- Plot 1: raw dual-axis ---
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(xs, mean_tuned, "o-", color="royalblue", lw=2, label="tuned-lens W−C at t*")
    ax1.plot(xs, mean_logit, "s--", color="cornflowerblue", lw=1.2, alpha=0.7, label="logit-lens W−C at t*")
    ax1.plot(xs_patch, mean_sens, "^-", color="darkorchid", lw=1.8, label="ŵ sensitivity at t* (proxy)")
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("layer (last point = final)")
    ax1.set_ylabel("readout score (W−C)", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")

    ax2 = ax1.twinx()
    ax2.bar(xs_patch, mean_patch, alpha=0.45, color="seagreen", label="patch Δ (C-swap at t*)")
    ax2.errorbar(xs_patch, mean_patch, yerr=sem_patch, fmt="none", ecolor="darkgreen", capsize=2)
    ax2.set_ylabel("causal Δ score (patched − baseline)", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("E4b follow-up: readout vs local sensitivity vs causal patch Δ")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "readout_vs_causality_raw.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: normalized overlay (shape comparison) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, normalize01(mean_tuned), "o-", lw=2, label="tuned-lens (norm)")
    ax.plot(xs_patch, normalize01(mean_sens), "^-", lw=2, label="ŵ sensitivity (norm)")
    ax.plot(xs_patch, normalize01(-mean_patch), "s-", lw=2, label="|patch Δ| (norm, inverted sign)")
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized 0–1")
    ax.set_title("Shape comparison: readout emergence vs causal leverage timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "readout_vs_causality_normalized.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: per-item spaghetti + means ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for t in tuned_all:
        axes[0].plot(xs, t, alpha=0.25, lw=1, color="royalblue")
    axes[0].plot(xs, mean_tuned, color="navy", lw=2.5, label="mean")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_title("Tuned-lens W−C at t*")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("score")

    for p in patch_all:
        axes[1].plot(xs_patch, p, alpha=0.25, lw=1, color="seagreen")
    axes[1].plot(xs_patch, mean_patch, color="darkgreen", lw=2.5, label="mean")
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_title("C-swap patch Δ at t*")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("Δ score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "readout_vs_causality_spaghetti.png", dpi=150)
    plt.close(fig)

    summary = {
        "mean_emergence_tuned": float(np.nanmean([r["emergence_tuned"] for r in rows if r["emergence_tuned"] is not None])),
        "mean_emergence_logit": float(np.nanmean([r["emergence_logit"] for r in rows if r["emergence_logit"] is not None])),
        "mean_causal_onset_layer": float(np.nanmean([r["causal_onset_layer"] for r in rows if r["causal_onset_layer"] is not None])),
        "mean_causal_peak_layer": float(np.mean([r["causal_peak_layer"] for r in rows])),
        "readout_before_causal_peak": bool(
            np.nanmean([r["emergence_tuned"] for r in rows if r["emergence_tuned"] is not None])
            < np.mean([r["causal_peak_layer"] for r in rows])
        ),
        "early_readout_low_causal": {
            f"L{L}": {
                "tuned": float(mean_tuned[L]),
                "w_sensitivity": float(mean_sens[L]),
                "patch_delta": float(mean_patch[L]),
            }
            for L in [3, 5, 7, 9, 11]
        },
        "jlens_note": (
            "Full Anthropic J-lens (github.com/anthropics/jacobian-lens) deferred: "
            "no pretrained gpt2 lens; fitting requires transformers>=5.5 and ~96 "
            "backward passes/prompt. ŵ sensitivity is a cheap directional proxy."
        ),
    }
    with open(OUT_DIR / "readout_vs_causality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(df_items.to_string(index=False))
    print("\nsummary:", json.dumps(summary, indent=2))
    print("Wrote plots to", OUT_DIR)


if __name__ == "__main__":
    main()

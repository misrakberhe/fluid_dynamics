"""E4b — Content patching at t* (when does the probe position itself decide?)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl

# Reuse E4 helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from E4_content_patching import (  # noqa: E402
    ITEMS,
    LANDING_LAYERS,
    PRIORITY_LAYERS,
    PatchResult,
    aggregate_results,
    get_positions,
    make_resid_patch_hooks,
    make_write_zero_hooks,
    run_baseline,
    run_with_hooks,
)

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E4b_outputs"
except NameError:
    OUT_DIR = Path("E4b_outputs")
OUT_DIR.mkdir(exist_ok=True)

EARLY_LAYERS = list(range(0, 5))
MID_LAYERS = list(range(5, 9))
LATE_LAYERS = list(range(9, 12))
ALL_LAYERS = list(range(12))


def run_item_interventions(model, item) -> list[PatchResult]:
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)

    w_tokens, _, W_pos, t_star, regions = get_positions(model, item, "W")
    c_tokens, _, _, _, _ = get_positions(model, item, "C")

    _, w_cache = model.run_with_cache(w_tokens)
    _, c_cache = model.run_with_cache(c_tokens)

    baseline = run_baseline(model, w_tokens, t_star, W_id, C_id)
    c_baseline = run_baseline(model, c_tokens, t_star, W_id, C_id)

    W_window = regions["W_window"]
    t_pos = [t_star]

    results: list[PatchResult] = []

    def record(name: str, patched: float, base: float = baseline):
        d = patched - base
        frac = d / base if abs(base) > 1e-6 else float("nan")
        results.append(PatchResult(item.name, name, base, patched, d, frac))

    # --- t* residual C-swap (necessity on W-prompt) ---
    for label, layers in [
        ("L0-4", EARLY_LAYERS),
        ("L5-8", MID_LAYERS),
        ("L9-11", LATE_LAYERS),
        ("L5-11", LANDING_LAYERS),
        ("L0-11", ALL_LAYERS),
    ]:
        hooks = make_resid_patch_hooks(c_cache, layers, t_pos)
        record(f"necessity_resid_tstar_{label}_Cswap", run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id))

    # --- t* residual W-patch (sufficiency on C-prompt) ---
    hooks = make_resid_patch_hooks(w_cache, LANDING_LAYERS, t_pos)
    record(
        "sufficiency_resid_tstar_L5-11_Wpatch",
        run_with_hooks(model, c_tokens, hooks, t_star, W_id, C_id),
        base=c_baseline,
    )

    # --- t* write ablation (compare to E4 write ablation at W) ---
    hooks = make_write_zero_hooks(LANDING_LAYERS, t_pos)
    record("ablate_writes_tstar_L5-11", run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id))

    hooks = make_write_zero_hooks(PRIORITY_LAYERS, t_pos)
    record("ablate_writes_tstar_L9-11", run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id))

    # --- E4 reference on same item (W_window band) for direct comparison ---
    hooks = make_resid_patch_hooks(c_cache, LANDING_LAYERS, W_window)
    record("ref_E4_necessity_Wwin_L5-11_Cswap", run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id))

    # --- Per-layer sweep at t* ---
    for L in range(model.cfg.n_layers):
        hooks = make_resid_patch_hooks(c_cache, [L], t_pos)
        record(f"necessity_resid_tstar_L{L}_Cswap", run_with_hooks(model, w_tokens, hooks, t_star, W_id, C_id))

    return results


def plot_tstar_layer_sweep(df: pd.DataFrame, fname: str):
    sweep = df[df["intervention"].str.match(r"necessity_resid_tstar_L\d+_Cswap$")]
    layers = sorted(int(n.split("L")[1].split("_")[0]) for n in sweep["intervention"].unique())
    mean_d, sem_d = [], []
    for L in layers:
        sub = sweep[sweep["intervention"] == f"necessity_resid_tstar_L{L}_Cswap"]["delta"]
        mean_d.append(sub.mean())
        sem_d.append(sub.std() / np.sqrt(len(sub)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layers, mean_d, yerr=sem_d, capsize=3, color="seagreen", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("layer (at position t*)")
    ax.set_ylabel("Δ score at t*")
    ax.set_title("E4b: per-layer C-swap at t* residual")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_comparison_sweep(e4b_df: pd.DataFrame, e4_df: pd.DataFrame | None, fname: str):
    """Overlay per-layer necessity: patch at t* vs patch at W_window."""

    def layer_curve(df, prefix: str):
        sweep = df[df["intervention"].str.match(prefix)]
        layers, means = [], []
        for L in range(12):
            name = sweep[sweep["intervention"].str.endswith(f"L{L}_Cswap")]
            if name.empty:
                continue
            layers.append(L)
            means.append(name["delta"].mean())
        return layers, means

    t_layers, t_means = layer_curve(e4b_df, r"necessity_resid_tstar_L\d+_Cswap$")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_layers, t_means, "o-", label="patch at t*", color="seagreen", lw=2)
    if e4_df is not None:
        w_layers, w_means = layer_curve(e4_df, r"necessity_resid_Wwin_L\d+_Cswap$")
        ax.plot(w_layers, w_means, "s-", label="patch at W_window (E4)", color="steelblue", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("layer patched")
    ax.set_ylabel("mean Δ score at t*")
    ax.set_title("E4 vs E4b: where does patching matter?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_band_bars(summary: pd.DataFrame, fname: str):
    keep = [
        "necessity_resid_tstar_L0-4_Cswap",
        "necessity_resid_tstar_L5-8_Cswap",
        "necessity_resid_tstar_L9-11_Cswap",
        "necessity_resid_tstar_L5-11_Cswap",
        "ref_E4_necessity_Wwin_L5-11_Cswap",
        "sufficiency_resid_tstar_L5-11_Wpatch",
        "ablate_writes_tstar_L5-11",
    ]
    sub = summary[summary["intervention"].isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean_delta"], yerr=sub["sem_delta"], capsize=3, color="mediumseagreen", alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sub["intervention"]], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("mean Δ score at t*")
    ax.set_title("E4b band interventions vs E4 W_window reference")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE, fold_ln=False)
    model.eval()

    all_results: list[PatchResult] = []
    for item in ITEMS:
        print("item:", item.name)
        all_results.extend(run_item_interventions(model, item))

    df = aggregate_results(all_results)
    df.to_csv(OUT_DIR / "all_interventions.csv", index=False)

    summary = (
        df.groupby("intervention")
        .agg(
            mean_baseline=("baseline", "mean"),
            mean_patched=("patched", "mean"),
            mean_delta=("delta", "mean"),
            sem_delta=("delta", "sem"),
            n=("delta", "count"),
        )
        .reset_index()
        .sort_values("mean_delta")
    )
    summary.to_csv(OUT_DIR / "intervention_summary.csv", index=False)

    key = summary[summary["intervention"].isin([
        "necessity_resid_tstar_L5-11_Cswap",
        "necessity_resid_tstar_L5-8_Cswap",
        "necessity_resid_tstar_L9-11_Cswap",
        "necessity_resid_tstar_L0-4_Cswap",
        "ref_E4_necessity_Wwin_L5-11_Cswap",
        "sufficiency_resid_tstar_L5-11_Wpatch",
        "ablate_writes_tstar_L5-11",
    ])]
    print(key.to_string(index=False))

    plot_tstar_layer_sweep(df, "per_layer_tstar_necessity.png")
    plot_band_bars(summary, "band_interventions.png")

    e4_path = Path(__file__).resolve().parent / "E4_outputs" / "all_interventions.csv"
    e4_df = pd.read_csv(e4_path) if e4_path.exists() else None
    plot_comparison_sweep(df, e4_df, "e4_vs_e4b_layer_sweep.png")

    # Per-layer print
    for L in range(12):
        row = summary[summary["intervention"] == f"necessity_resid_tstar_L{L}_Cswap"]
        if not row.empty:
            print(f"t* L{L}: delta={row['mean_delta'].iloc[0]:.3f}")

    t511 = summary.loc[summary["intervention"] == "necessity_resid_tstar_L5-11_Cswap", "mean_delta"].iloc[0]
    w511 = summary.loc[summary["intervention"] == "ref_E4_necessity_Wwin_L5-11_Cswap", "mean_delta"].iloc[0]
    t_late = summary.loc[summary["intervention"] == "necessity_resid_tstar_L9-11_Cswap", "mean_delta"].iloc[0]
    t_early = summary.loc[summary["intervention"] == "necessity_resid_tstar_L0-4_Cswap", "mean_delta"].iloc[0]

    verdict = {
        "tstar_resid_swap_L5-11_delta": float(t511),
        "W_window_resid_swap_L5-11_delta": float(w511),
        "tstar_late_band_L9-11_delta": float(t_late),
        "tstar_early_band_L0-4_delta": float(t_early),
        "decision_mostly_at_tstar_locally": bool(abs(t_late) > abs(w511) * 0.5),
        "source_W_still_necessary": bool(w511 < -1.0),
    }
    with open(OUT_DIR / "e4b_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

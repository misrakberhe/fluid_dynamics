"""Qwen replication of forced answer anchoring (Phase 1+).

Loads Qwen3.5 via TransformerLens 3 TransformerBridge, audits tokenization,
and runs behavioral / causal experiments ported from E4_content_patching.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformer_lens.model_bridge import TransformerBridge

from E4_content_patching import (
    ITEMS as GPT2_ITEMS,
    Item,
    build_C_prompt,
    build_W_prompt,
    make_resid_patch_hooks,
    score_at_tstar,
)

torch.set_grad_enabled(False)

OUT_DIR = Path(__file__).resolve().parent / "qwen_replication_outputs"
GPT2_BEHAVIOR_REF = Path(__file__).resolve().parent / "forced_W_vs_C_outputs" / "summary.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_CPU = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_GPU = "Qwen/Qwen3.5-4B"

# Single-token answers on Qwen (no leading space; digits 0-9 only).
# GPT-2 bank uses multi-digit strings like " 25" — all fail Qwen single-token audit.
QWEN_ITEMS = [
    Item("ops_4+5", 4, 5, "8", "9"),
    Item("ops_1+8", 1, 8, "7", "9"),
    Item("ops_2+7", 2, 7, "8", "9"),
    Item("ops_6+3", 6, 3, "8", "9"),
    Item("ops_1+1", 1, 1, "3", "2"),
    Item("ops_2+3", 2, 3, "4", "5"),
    Item("ops_5+4", 5, 4, "8", "9"),
    Item("ops_1+6", 1, 6, "6", "7"),
]


def default_model_id() -> str:
    return DEFAULT_MODEL_GPU if DEVICE == "cuda" else DEFAULT_MODEL_CPU


def load_model(model_id: str | None = None) -> TransformerBridge:
    model_id = model_id or default_model_id()
    print(f"Loading {model_id} on {DEVICE}...")
    bridge = TransformerBridge.boot_transformers(model_id, device=DEVICE)
    bridge.eval()
    return bridge


def is_single_token(model, text: str) -> bool:
    try:
        model.to_single_token(text)
        return True
    except Exception:
        return False


def str_token_list(model, text: str) -> list[str]:
    return model.to_str_tokens(text, prepend_bos=False)


# --- Qwen-aware position helpers (GPT-2 uses exact " =" tokens) ---


def find_eq_positions(str_toks: list[str]) -> list[int]:
    return [i for i, tok in enumerate(str_toks) if "=" in tok]


def find_t_star(str_toks: list[str]) -> int:
    eq = find_eq_positions(str_toks)
    if not eq:
        raise ValueError("no '=' token in prompt")
    return eq[-1]


def find_impulse_pos(str_toks: list[str]) -> int:
    """First token after the first '=' (forced-answer slot)."""
    return find_eq_positions(str_toks)[0] + 1


def tag_regions(str_toks: list[str], impulse_pos: int, t_star: int) -> dict[str, list[int]]:
    ops2_pos = list(range(t_star - 3, t_star))
    rev_start = next(j for j, tok in enumerate(str_toks) if tok.strip() == "Wait")
    rev_end = ops2_pos[0] - 1
    revision = list(range(rev_start, rev_end + 1))
    W_window = list(range(max(0, impulse_pos - 2), min(len(str_toks), impulse_pos + 3)))
    return {
        "W": [impulse_pos],
        "W_window": W_window,
        "revision": revision,
        "ops2": ops2_pos,
    }


def get_positions(
    model,
    item: Item,
    variant: str = "W",
) -> tuple[torch.Tensor, list[str], int, int, dict[str, list[int]]]:
    prompt = build_W_prompt(item) if variant == "W" else build_C_prompt(item)
    tokens = model.to_tokens(prompt)
    str_toks = model.to_str_tokens(prompt)
    impulse_pos = find_impulse_pos(str_toks)
    t_star = find_t_star(str_toks)
    regions = tag_regions(str_toks, impulse_pos, t_star)
    return tokens, str_toks, impulse_pos, t_star, regions


def run_baseline(model, tokens, t_star: int, W_id: int, C_id: int) -> float:
    logits = model(tokens)
    return score_at_tstar(model, logits, t_star, W_id, C_id)


def run_with_hooks(model, tokens, hooks, t_star: int, W_id: int, C_id: int) -> float:
    with model.hooks(hooks):
        logits = model(tokens)
    return score_at_tstar(model, logits, t_star, W_id, C_id)


def run_with_cache(model, tokens):
    return model.run_with_cache(tokens)


def audit_item(model, item: Item, bank: str) -> dict:
    prompt_w = build_W_prompt(item)
    str_toks = model.to_str_tokens(prompt_w)
    eq_positions = find_eq_positions(str_toks)
    impulse_pos = find_impulse_pos(str_toks)
    t_star = find_t_star(str_toks)
    regions = tag_regions(str_toks, impulse_pos, t_star)

    w_single = is_single_token(model, item.W_str)
    c_single = is_single_token(model, item.C_str)
    w_toks = str_token_list(model, item.W_str)
    c_toks = str_token_list(model, item.C_str)
    correct = item.a + item.b

    pass_audit = w_single and c_single and item.C_str.strip() == str(correct)

    return {
        "bank": bank,
        "item": item.name,
        "a": item.a,
        "b": item.b,
        "correct_sum": correct,
        "W_str": item.W_str,
        "C_str": item.C_str,
        "W_single_token": w_single,
        "C_single_token": c_single,
        "W_token_strs": "|".join(w_toks),
        "C_token_strs": "|".join(c_toks),
        "C_is_correct_sum": item.C_str.strip() == str(correct),
        "pass_audit": pass_audit,
        "prompt_W": prompt_w,
        "n_tokens": len(str_toks),
        "eq_positions": "|".join(map(str, eq_positions)),
        "impulse_pos": impulse_pos,
        "impulse_token": str_toks[impulse_pos],
        "t_star": t_star,
        "t_star_token": str_toks[t_star],
        "W_window": "|".join(map(str, regions["W_window"])),
        "revision": "|".join(map(str, regions["revision"])),
        "ops2": "|".join(map(str, regions["ops2"])),
    }


def smoke_test(model, item: Item | None = None) -> dict:
    """One-item forward pass + logits at t*; indices must match token_audit.csv."""
    audit_df = pd.read_csv(OUT_DIR / "token_audit.csv")
    if item is None:
        passing = audit_df[(audit_df["pass_audit"]) & (audit_df["bank"] == "qwen")]
        if passing.empty:
            raise RuntimeError("No items pass token audit; cannot run smoke test")
        row = passing.iloc[0]
        item = next(i for i in QWEN_ITEMS if i.name == row["item"])

    audit_row = audit_df[(audit_df["item"] == item.name) & (audit_df["bank"] == "qwen")].iloc[0]
    tokens, str_toks, impulse_pos, t_star, regions = get_positions(model, item, "W")
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)

    logits = model(tokens)
    assert torch.isfinite(logits).all(), "logits contain non-finite values"
    score = score_at_tstar(model, logits, t_star, W_id, C_id)
    top1_id = int(logits[0, t_star].argmax())
    top1 = model.to_string(top1_id)

    assert impulse_pos == int(audit_row["impulse_pos"]), "impulse_pos mismatch vs audit"
    assert t_star == int(audit_row["t_star"]), "t_star mismatch vs audit"

    _, cache = run_with_cache(model, tokens)
    hook_key = "blocks.0.hook_resid_post"
    assert hook_key in cache, f"missing cache key {hook_key}"

    result = {
        "item": item.name,
        "model": model.cfg.model_name if hasattr(model.cfg, "model_name") else "qwen",
        "n_layers": model.cfg.n_layers,
        "device": DEVICE,
        "prompt": build_W_prompt(item),
        "n_tokens": len(str_toks),
        "impulse_pos": impulse_pos,
        "impulse_token": str_toks[impulse_pos],
        "t_star": t_star,
        "score_W_minus_C": score,
        "top1_at_tstar": top1,
        "logits_finite": True,
        "cache_hook_ok": True,
        "indices_match_audit": True,
    }
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "smoke_test.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Smoke test OK:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"Wrote {OUT_DIR / 'smoke_test.json'}")
    return result


def evaluate_behavior(model, item: Item, variant: str) -> dict:
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
    bank_label = item.W_str if variant == "W" else item.C_str
    return {
        "item": item.name,
        "variant": variant,
        "prompt": prompt,
        "t_star": t_star,
        "impulse_pos": impulse_pos,
        "impulse_token": impulse_tok,
        "score_W_minus_C": score,
        "abs_score": abs(score),
        "top1": top1,
        "top1_is_bank_W": top1 == item.W_str,
        "top1_is_bank_C": top1 == item.C_str,
        "top1_is_bank_impulse": top1 == bank_label,
        "top1_matches_impulse": top1 == impulse_tok,
    }


def run_behavior(model, items: list[Item] | None = None, include_forced_c: bool = True) -> pd.DataFrame:
    items = items or QWEN_ITEMS
    rows = []
    variants = ("W", "C") if include_forced_c else ("W",)
    for item in items:
        print("behavior:", item.name)
        for variant in variants:
            rows.append(evaluate_behavior(model, item, variant))
    behavior = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    behavior.to_csv(OUT_DIR / "behavior.csv", index=False)
    print(f"Wrote {OUT_DIR / 'behavior.csv'}")
    return behavior


def summarize_behavior(behavior: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for variant in behavior["variant"].unique():
        sub = behavior[behavior["variant"] == variant]
        summary_rows.append({
            "variant": variant,
            "n_items": len(sub),
            "mean_score_W_minus_C": sub["score_W_minus_C"].mean(),
            "sem_score_W_minus_C": sub["score_W_minus_C"].sem(),
            "mean_abs_score": sub["abs_score"].mean(),
            "frac_top1_impulse": sub["top1_matches_impulse"].mean(),
            "frac_top1_bank_W": sub["top1_is_bank_W"].mean(),
            "frac_top1_bank_C": sub["top1_is_bank_C"].mean(),
        })
    summary = pd.DataFrame(summary_rows)

    if "W" in summary["variant"].values:
        w_sub = behavior[behavior["variant"] == "W"]
        n_top1_w = int(w_sub["top1_is_bank_W"].sum())
        summary = summary.copy()
        summary["g1_mean_score_pass"] = summary.apply(
            lambda r: r["mean_score_W_minus_C"] > 1.0 if r["variant"] == "W" else None,
            axis=1,
        )
        summary["g1_top1_w_count"] = summary.apply(
            lambda r: n_top1_w if r["variant"] == "W" else None,
            axis=1,
        )
        summary["g1_top1_w_pass"] = summary.apply(
            lambda r: n_top1_w >= max(5, int(0.75 * len(w_sub))) if r["variant"] == "W" else None,
            axis=1,
        )

    summary.to_csv(OUT_DIR / "behavior_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'behavior_summary.csv'}")
    print(summary.to_string(index=False))
    return summary


def evaluate_g1(behavior: pd.DataFrame) -> dict:
    w = behavior[behavior["variant"] == "W"]
    mean_score = float(w["score_W_minus_C"].mean())
    n_top1_w = int(w["top1_is_bank_W"].sum())
    n_items = len(w)
    threshold = max(5, int(0.75 * n_items))
    g1_pass = mean_score > 1.0 and n_top1_w >= threshold
    return {
        "g1_behavior_pass": g1_pass,
        "mean_score_forced_W": mean_score,
        "frac_top1_bank_W": float(w["top1_is_bank_W"].mean()),
        "n_top1_bank_W": n_top1_w,
        "n_items": n_items,
        "g1_mean_score_threshold": 1.0,
        "g1_top1_min_count": threshold,
        "note": (
            "Qwen may predict whitespace at t* before the answer digit; "
            "see behavior.csv top1 vs top1_is_bank_W."
        ),
    }


def load_gpt2_behavior_reference() -> dict:
    if not GPT2_BEHAVIOR_REF.exists():
        return {
            "mean_score_forced_W": 2.71,
            "mean_score_forced_C": -2.68,
            "frac_top1_impulse_forced_W": 1.0,
            "frac_top1_impulse_forced_C": 1.0,
            "source": "plan_fallback",
        }
    row = pd.read_csv(GPT2_BEHAVIOR_REF).iloc[0]
    return {
        "mean_score_forced_W": float(row["mean_score_forced_W"]),
        "mean_score_forced_C": float(row["mean_score_forced_C"]),
        "frac_top1_impulse_forced_W": float(row["frac_top1_impulse_forced_W"]),
        "frac_top1_impulse_forced_C": float(row["frac_top1_impulse_forced_C"]),
        "source": str(GPT2_BEHAVIOR_REF),
    }


def plot_gpt2_vs_qwen(behavior: pd.DataFrame):
    gpt2 = load_gpt2_behavior_reference()
    qwen_w = behavior[behavior["variant"] == "W"]
    qwen_c = behavior[behavior["variant"] == "C"]

    models = ["GPT-2", "Qwen3.5"]
    w_scores = [gpt2["mean_score_forced_W"], qwen_w["score_W_minus_C"].mean()]
    c_scores = [gpt2["mean_score_forced_C"], qwen_c["score_W_minus_C"].mean()] if len(qwen_c) else [gpt2["mean_score_forced_C"], np.nan]
    w_frac = [gpt2["frac_top1_impulse_forced_W"], qwen_w["top1_matches_impulse"].mean()]
    c_frac = [gpt2["frac_top1_impulse_forced_C"], qwen_c["top1_matches_impulse"].mean()] if len(qwen_c) else [gpt2["frac_top1_impulse_forced_C"], np.nan]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width / 2, w_scores, width, label="forced W", color="#e74c3c", alpha=0.85)
    axes[0].bar(x + width / 2, c_scores, width, label="forced C", color="#2ecc71", alpha=0.85)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("mean score @ t* (logit W − logit C)")
    axes[0].set_title("Behavior: GPT-2 vs Qwen")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, w_frac, width, label="forced W", color="#e74c3c", alpha=0.85)
    axes[1].bar(x + width / 2, c_frac, width, label="forced C", color="#2ecc71", alpha=0.85)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("frac top-1 = impulse token")
    axes[1].set_title("Persistence @ t*")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavior_gpt2_vs_qwen.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'behavior_gpt2_vs_qwen.png'}")


def run_behavior_phase(model, include_forced_c: bool = True) -> dict:
    behavior = run_behavior(model, include_forced_c=include_forced_c)
    summary = summarize_behavior(behavior)
    plot_gpt2_vs_qwen(behavior)
    g1 = evaluate_g1(behavior)
    gpt2_ref = load_gpt2_behavior_reference()

    if "C" in behavior["variant"].values:
        c = behavior[behavior["variant"] == "C"]
        g1["mean_score_forced_C"] = float(c["score_W_minus_C"].mean())
        g1["frac_top1_impulse_forced_C"] = float(c["top1_matches_impulse"].mean())

    verdict = {
        **g1,
        "gpt2_reference": gpt2_ref,
        "phase": 2,
    }
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"G1 behavior gate: {'PASS' if g1['g1_behavior_pass'] else 'FAIL'}")
    print(f"Wrote {OUT_DIR / 'verdict.json'}")
    return verdict


def main():
    parser = argparse.ArgumentParser(description="Qwen anchoring replication")
    parser.add_argument(
        "command",
        choices=["token-audit", "smoke-test", "behavior", "all", "phase1", "phase2"],
        help="Experiment subcommand",
    )
    parser.add_argument("--model", default=None, help="HuggingFace model id")
    parser.add_argument("--item", default=None, help="Item name for smoke-test")
    parser.add_argument(
        "--no-forced-c",
        action="store_true",
        help="Skip forced-C condition in behavior (default: include both W and C)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    model = load_model(args.model)

    if args.command in ("token-audit", "all", "phase1"):
        gpt2_df = pd.DataFrame([audit_item(model, it, "gpt2") for it in GPT2_ITEMS])
        qwen_df = pd.DataFrame([audit_item(model, it, "qwen") for it in QWEN_ITEMS])
        combined = pd.concat([gpt2_df, qwen_df], ignore_index=True)
        combined.to_csv(OUT_DIR / "token_audit.csv", index=False)
        n_pass = int(combined["pass_audit"].sum())
        print(f"Token audit: {n_pass}/{len(combined)} items pass")
        print(f"  gpt2 bank on Qwen: {int(gpt2_df['pass_audit'].sum())}/{len(gpt2_df)}")
        print(f"  qwen bank: {int(qwen_df['pass_audit'].sum())}/{len(qwen_df)}")
        print(f"Wrote {OUT_DIR / 'token_audit.csv'}")

    if args.command in ("smoke-test", "all", "phase1"):
        item = next((i for i in QWEN_ITEMS if i.name == args.item), None) if args.item else None
        smoke_test(model, item=item)

    if args.command in ("behavior", "all", "phase2"):
        run_behavior_phase(model, include_forced_c=not args.no_forced_c)


if __name__ == "__main__":
    main()

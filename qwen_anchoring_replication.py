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


def score_at_pos(logits, pos: int, W_id: int, C_id: int) -> float:
    v = logits[0, pos]
    return float(v[W_id] - v[C_id])


def get_scoring_setup(model, item: Item, variant: str) -> dict:
    """Base prompt tokens/cache + extended score_tokens at answer_pos."""
    prompt = build_W_prompt(item) if variant == "W" else build_C_prompt(item)
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)
    impulse_pos = find_impulse_pos(str_toks)
    regions = tag_regions(str_toks, impulse_pos, t_star)
    tokens = model.to_tokens(prompt)

    logits0 = model(tokens)
    top1_at_tstar = model.to_string([int(logits0[0, t_star].argmax())])
    if top1_at_tstar in (item.W_str, item.C_str):
        score_prompt = prompt
        score_tokens = tokens
        answer_pos = t_star
        answer_prefix = ""
    else:
        answer_prefix = top1_at_tstar
        score_prompt = prompt + answer_prefix
        score_tokens = model.to_tokens(score_prompt)
        answer_pos = len(model.to_str_tokens(score_prompt)) - 1

    return {
        "prompt": prompt,
        "score_prompt": score_prompt,
        "tokens": tokens,
        "score_tokens": score_tokens,
        "str_toks": str_toks,
        "t_star": t_star,
        "answer_pos": answer_pos,
        "answer_prefix": answer_prefix,
        "impulse_pos": impulse_pos,
        "regions": regions,
    }


def score_with_hooks(
    model,
    score_tokens: torch.Tensor,
    answer_pos: int,
    W_id: int,
    C_id: int,
    hooks: list | None = None,
) -> float:
    if hooks:
        with model.hooks(hooks):
            logits = model(score_tokens)
    else:
        logits = model(score_tokens)
    return score_at_pos(logits, answer_pos, W_id, C_id)


def landing_layers(n_layers: int, frac_start: float = 0.25, frac_end: float = 0.75) -> list[int]:
    """Mid-depth band (~25–75% of layers), matching Phase 3 plan."""
    start = max(0, int(n_layers * frac_start))
    end = max(start + 1, int(n_layers * frac_end))
    return list(range(start, end))


def random_control_positions(str_toks: list[str], t_star: int, W_window: list[int]) -> list[int]:
    """Distance-matched random positions (E4 protocol)."""
    width = len(W_window)
    center = t_star - 3
    rand_pos = list(
        range(max(1, center - width // 2), min(len(str_toks), center - width // 2 + width))
    )
    return [p for p in rand_pos if p != t_star][:width]


def run_item_causal(
    model,
    item: Item,
    layers: list[int],
    layer_label: str,
) -> list[dict]:
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)

    w = get_scoring_setup(model, item, "W")
    c = get_scoring_setup(model, item, "C")

    _, w_cache = model.run_with_cache(w["tokens"])
    _, c_cache = model.run_with_cache(c["tokens"])

    baseline = score_with_hooks(model, w["score_tokens"], w["answer_pos"], W_id, C_id)
    W_window = w["regions"]["W_window"]
    rand_pos = random_control_positions(w["str_toks"], w["t_star"], W_window)

    rows = []

    def record(intervention: str, patched: float):
        delta = patched - baseline
        frac = delta / baseline if abs(baseline) > 1e-6 else float("nan")
        rows.append({
            "item": item.name,
            "intervention": intervention,
            "layer_band": layer_label,
            "layers": "|".join(map(str, layers)),
            "baseline": baseline,
            "patched": patched,
            "delta": delta,
            "frac_of_baseline": frac,
            "answer_pos": w["answer_pos"],
            "answer_prefix": w["answer_prefix"],
            "W_window": "|".join(map(str, W_window)),
            "rand_pos": "|".join(map(str, rand_pos)),
        })

    hooks = make_resid_patch_hooks(c_cache, layers, W_window)
    record(f"necessity_resid_Wwin_{layer_label}_Cswap", score_with_hooks(
        model, w["score_tokens"], w["answer_pos"], W_id, C_id, hooks
    ))

    hooks = make_resid_patch_hooks(c_cache, layers, rand_pos)
    record(f"control_resid_randpos_{layer_label}_Cswap", score_with_hooks(
        model, w["score_tokens"], w["answer_pos"], W_id, C_id, hooks
    ))

    return rows


def summarize_causal(causal: pd.DataFrame) -> pd.DataFrame:
    summary = (
        causal.groupby("intervention")
        .agg(
            mean_baseline=("baseline", "mean"),
            mean_patched=("patched", "mean"),
            mean_delta=("delta", "mean"),
            sem_delta=("delta", "sem"),
            mean_frac=("frac_of_baseline", "mean"),
            n=("delta", "count"),
        )
        .reset_index()
        .sort_values("mean_delta")
    )
    summary.to_csv(OUT_DIR / "causal_summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'causal_summary.csv'}")
    print(summary.to_string(index=False))
    return summary


def plot_causal(causal: pd.DataFrame, layer_label: str):
    summary = (
        causal.groupby("intervention")
        .agg(mean_delta=("delta", "mean"), sem_delta=("delta", "sem"))
    )
    order = [
        f"necessity_resid_Wwin_{layer_label}_Cswap",
        f"control_resid_randpos_{layer_label}_Cswap",
    ]
    labels = ["W_window C-swap", "rand-pos control"]
    colors = ["#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(order))
    means = [summary.loc[i, "mean_delta"] if i in summary.index else 0 for i in order]
    sems = [summary.loc[i, "sem_delta"] if i in summary.index else 0 for i in order]
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean Δ score @ answer_pos")
    ax.set_title(f"Phase 3 causal (Qwen, {layer_label})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "causal_interventions.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'causal_interventions.png'}")


def evaluate_g2(causal: pd.DataFrame, layer_label: str) -> dict:
    cswap_name = f"necessity_resid_Wwin_{layer_label}_Cswap"
    rand_name = f"control_resid_randpos_{layer_label}_Cswap"
    cswap = causal[causal["intervention"] == cswap_name]["delta"]
    rand = causal[causal["intervention"] == rand_name]["delta"]
    mean_cswap = float(cswap.mean())
    mean_rand = float(rand.mean())
    g2_pass = mean_cswap < -1.0 and abs(mean_cswap) > abs(mean_rand) * 2
    return {
        "g2_causal_pass": g2_pass,
        "mean_delta_Wwin_Cswap": mean_cswap,
        "mean_delta_rand_control": mean_rand,
        "g2_cswap_threshold": -1.0,
        "gpt2_reference_Wwin_Cswap_delta": -4.94,
        "scoring_position": "answer_pos (after greedy prefix following t*)",
    }


def load_verdict() -> dict:
    path = OUT_DIR / "verdict.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run_causal_phase(
    model,
    items: list[Item] | None = None,
    frac_start: float = 0.25,
    frac_end: float = 0.75,
) -> dict:
    items = items or QWEN_ITEMS
    n_layers = model.cfg.n_layers
    layers = landing_layers(n_layers, frac_start, frac_end)
    layer_label = f"L{layers[0]}-{layers[-1]}"

    rows = []
    for item in items:
        print("causal:", item.name)
        rows.extend(run_item_causal(model, item, layers, layer_label))

    causal = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    causal.to_csv(OUT_DIR / "causal.csv", index=False)
    print(f"Wrote {OUT_DIR / 'causal.csv'}")

    summary = summarize_causal(causal)
    plot_causal(causal, layer_label)
    g2 = evaluate_g2(causal, layer_label)

    verdict = load_verdict()
    verdict.update(g2)
    verdict["phase"] = 3
    verdict["causal_layer_band"] = layer_label
    verdict["n_layers"] = n_layers
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"G2 causal gate: {'PASS' if g2['g2_causal_pass'] else 'FAIL'}")
    print(f"Wrote {OUT_DIR / 'verdict.json'}")
    return verdict


def evaluate_behavior(model, item: Item, variant: str) -> dict:
    prompt = build_W_prompt(item) if variant == "W" else build_C_prompt(item)
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)
    impulse_pos = find_impulse_pos(str_toks)
    impulse_tok = str_toks[impulse_pos]
    bank_label = item.W_str if variant == "W" else item.C_str
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)

    logits0 = model(model.to_tokens(prompt))
    top1_at_tstar = model.to_string([int(logits0[0, t_star].argmax())])
    score_at_tstar = score_at_pos(logits0, t_star, W_id, C_id)

    # Qwen: t* usually predicts whitespace; answer digit is at the next step.
    if top1_at_tstar in (item.W_str, item.C_str):
        logits_answer = logits0
        answer_pos = t_star
        answer_prefix = ""
    else:
        answer_prefix = top1_at_tstar
        logits_answer = model(model.to_tokens(prompt + answer_prefix))
        answer_pos = len(model.to_str_tokens(prompt + answer_prefix)) - 1

    score = score_at_pos(logits_answer, answer_pos, W_id, C_id)
    top1 = model.to_string([int(logits_answer[0, answer_pos].argmax())])

    return {
        "item": item.name,
        "variant": variant,
        "prompt": prompt,
        "t_star": t_star,
        "answer_pos": answer_pos,
        "answer_prefix": answer_prefix,
        "impulse_pos": impulse_pos,
        "impulse_token": impulse_tok,
        "score_W_minus_C_at_tstar": score_at_tstar,
        "score_W_minus_C": score,
        "abs_score": abs(score),
        "top1_at_tstar": top1_at_tstar,
        "top1": top1,
        "top1_is_bank_W": top1 == item.W_str,
        "top1_is_bank_C": top1 == item.C_str,
        "top1_is_bank_impulse": top1 == bank_label,
        "top1_matches_impulse": top1 == impulse_tok,
        "top1_at_tstar_matches_impulse": top1_at_tstar == impulse_tok,
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
            "mean_score_W_minus_C_at_tstar": sub["score_W_minus_C_at_tstar"].mean(),
            "mean_abs_score": sub["abs_score"].mean(),
            "frac_top1_impulse": sub["top1_matches_impulse"].mean(),
            "frac_top1_bank_W": sub["top1_is_bank_W"].mean(),
            "frac_top1_bank_C": sub["top1_is_bank_C"].mean(),
            "frac_top1_whitespace_at_tstar": (sub["top1_at_tstar"].str.strip() == "").mean(),
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
        "scoring_position": "answer_pos (after greedy prefix following t*; usually whitespace on Qwen)",
        "note": (
            "Primary score/top1 use answer_pos. score_W_minus_C_at_tstar / top1_at_tstar kept for comparison. "
            "GPT-2 scores at t* directly (digit is top-1 there)."
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
    axes[0].set_ylabel("mean score @ answer pos (logit W − logit C)")
    axes[0].set_title("Behavior: GPT-2 @ t* vs Qwen @ answer pos")
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


def plot_gpt2_vs_qwen_causal(causal: pd.DataFrame, layer_label: str):
    """Side-by-side GPT-2 vs Qwen W_window C-swap Δ (Phase 4 application figure)."""
    cswap_name = f"necessity_resid_Wwin_{layer_label}_Cswap"
    qwen_row = causal[causal["intervention"] == cswap_name]
    if qwen_row.empty:
        print(f"Warning: no {cswap_name} in causal.csv; skipping replication figure")
        return

    qwen_delta = float(qwen_row["delta"].mean())
    qwen_sem = float(qwen_row["delta"].sem())
    gpt2_delta = -4.94  # forced_W_vs_C / E4 reference

    models = ["GPT-2", "Qwen3.5"]
    means = [gpt2_delta, qwen_delta]
    sems = [0.0, qwen_sem]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(models))
    colors = ["#3498db", "#e74c3c"]
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("mean Δ score (W_window C-swap)")
    ax.set_title("Causal: GPT-2 L5–11 vs Qwen L8–23")
    fig.tight_layout()
    out = OUT_DIR / "replication_causal_gpt2_vs_qwen.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def run_phase4_packaging():
    """Phase 4: regenerate figures from saved outputs (no GPU / model load)."""
    behavior_path = OUT_DIR / "behavior.csv"
    causal_path = OUT_DIR / "causal.csv"
    verdict_path = OUT_DIR / "verdict.json"

    if not behavior_path.exists():
        raise FileNotFoundError(f"Missing {behavior_path}; run phase2 first")

    behavior = pd.read_csv(behavior_path)
    plot_gpt2_vs_qwen(behavior)

    layer_label = "L8-23"
    if verdict_path.exists():
        with open(verdict_path) as f:
            verdict = json.load(f)
        layer_label = verdict.get("causal_layer_band", layer_label)
        g1 = "PASS" if verdict.get("g1_behavior_pass") else "FAIL"
        g2 = "PASS" if verdict.get("g2_causal_pass") else "FAIL"
        print(f"G1 behavior: {g1} | G2 causal: {g2}")

    if causal_path.exists():
        causal = pd.read_csv(causal_path)
        plot_gpt2_vs_qwen_causal(causal, layer_label)
    else:
        print(f"Warning: {causal_path} missing; skip causal comparison figure")

    print("Phase 4 packaging artifacts:")
    for name in [
        "qwen_anchoring_replication_session_summary.md",
        "behavior_gpt2_vs_qwen.png",
        "causal_interventions.png",
        "replication_causal_gpt2_vs_qwen.png",
    ]:
        path = Path(__file__).resolve().parent / name if name.endswith(".md") else OUT_DIR / name
        status = "ok" if path.exists() else "missing"
        print(f"  [{status}] {path.name}")


def run_behavior_phase(model, include_forced_c: bool = True) -> dict:
    behavior = run_behavior(model, include_forced_c=include_forced_c)
    summarize_behavior(behavior)
    plot_gpt2_vs_qwen(behavior)
    g1 = evaluate_g1(behavior)
    gpt2_ref = load_gpt2_behavior_reference()

    if "C" in behavior["variant"].values:
        c = behavior[behavior["variant"] == "C"]
        g1["mean_score_forced_C"] = float(c["score_W_minus_C"].mean())
        g1["frac_top1_impulse_forced_C"] = float(c["top1_matches_impulse"].mean())

    verdict = load_verdict()
    verdict.update(g1)
    verdict["gpt2_reference"] = gpt2_ref
    verdict["phase"] = max(verdict.get("phase", 0), 2)
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"G1 behavior gate: {'PASS' if g1['g1_behavior_pass'] else 'FAIL'}")
    print(f"Wrote {OUT_DIR / 'verdict.json'}")
    return verdict


def main():
    parser = argparse.ArgumentParser(description="Qwen anchoring replication")
    parser.add_argument(
        "command",
        choices=[
            "token-audit",
            "smoke-test",
            "behavior",
            "causal",
            "all",
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "package",
        ],
        help="Experiment subcommand",
    )
    parser.add_argument("--model", default=None, help="HuggingFace model id")
    parser.add_argument("--item", default=None, help="Item name for smoke-test")
    parser.add_argument(
        "--layer-frac-start",
        type=float,
        default=0.25,
        help="Causal patch band start fraction of depth (default 0.25)",
    )
    parser.add_argument(
        "--layer-frac-end",
        type=float,
        default=0.75,
        help="Causal patch band end fraction of depth (default 0.75)",
    )
    parser.add_argument(
        "--no-forced-c",
        action="store_true",
        help="Skip forced-C condition in behavior (default: include both W and C)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if args.command in ("phase4", "package"):
        run_phase4_packaging()
        return

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

    if args.command in ("causal", "phase3"):
        run_causal_phase(
            model,
            frac_start=args.layer_frac_start,
            frac_end=args.layer_frac_end,
        )


if __name__ == "__main__":
    main()

"""Qwen replication of forced answer anchoring (Phase 1+).

Loads Qwen3.5 via TransformerLens 3 TransformerBridge, audits tokenization,
and runs behavioral / causal experiments ported from E4_content_patching.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Qwen anchoring replication")
    parser.add_argument(
        "command",
        choices=["token-audit", "smoke-test", "all"],
        help="Phase 1 subcommand",
    )
    parser.add_argument("--model", default=None, help="HuggingFace model id")
    parser.add_argument("--item", default=None, help="Item name for smoke-test")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    model = load_model(args.model)

    if args.command in ("token-audit", "all"):
        gpt2_df = pd.DataFrame([audit_item(model, it, "gpt2") for it in GPT2_ITEMS])
        qwen_df = pd.DataFrame([audit_item(model, it, "qwen") for it in QWEN_ITEMS])
        combined = pd.concat([gpt2_df, qwen_df], ignore_index=True)
        OUT_DIR.mkdir(exist_ok=True)
        combined.to_csv(OUT_DIR / "token_audit.csv", index=False)
        n_pass = int(combined["pass_audit"].sum())
        print(f"Token audit: {n_pass}/{len(combined)} items pass")
        print(f"  gpt2 bank on Qwen: {int(gpt2_df['pass_audit'].sum())}/{len(gpt2_df)}")
        print(f"  qwen bank: {int(qwen_df['pass_audit'].sum())}/{len(qwen_df)}")
        print(f"Wrote {OUT_DIR / 'token_audit.csv'}")

    if args.command in ("smoke-test", "all"):
        item = next((i for i in QWEN_ITEMS if i.name == args.item), None) if args.item else None
        smoke_test(model, item=item)


if __name__ == "__main__":
    main()

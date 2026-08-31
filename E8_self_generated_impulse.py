"""E8 — Self-generated vs forced impulse (input-copying vs commitment)."""

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
from E4_content_patching import ITEMS, CUE  # noqa: E402

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    OUT_DIR = Path(__file__).resolve().parent / "E8_outputs"
except NameError:
    OUT_DIR = Path("E8_outputs")
OUT_DIR.mkdir(exist_ok=True)

MAX_ANSWER_TOKENS = 4
STOP_TOKENS = (".", "\n")


@dataclass
class GenResult:
    prefix: str
    impulse_str: str
    first_token: str
    full_prefix: str


def strip_bos(s: str) -> str:
    return s.replace("<|endoftext|>", "")


def find_t_star(str_toks: list[str]) -> int:
    return [i for i, t in enumerate(str_toks) if t == " ="][-1]


def build_prefix(item) -> str:
    return f"{item.a} + {item.b} ="


def greedy_generate_impulse(model, prefix: str) -> GenResult:
    tokens = model.to_tokens(prefix)
    for _ in range(MAX_ANSWER_TOKENS):
        logits = model(tokens)
        nid = int(logits[0, -1].argmax())
        tokens = torch.cat([tokens, torch.tensor([[nid]], device=tokens.device)], dim=1)
        if model.to_string([nid]) in STOP_TOKENS:
            break
    full = strip_bos(model.to_string(tokens[0]))
    str_toks = model.to_str_tokens(full)
    eq_idx = next(i for i, t in enumerate(str_toks) if t == " =")
    first_token = str_toks[eq_idx + 1] if eq_idx + 1 < len(str_toks) else ""
    impulse_toks = str_toks[eq_idx + 1 :]
    impulse_str = "".join(impulse_toks)
    return GenResult(prefix=prefix, impulse_str=impulse_str, first_token=first_token, full_prefix=full)


def build_prompts(item, gen: GenResult) -> dict[str, str]:
    suffix = f"{CUE} {item.a} + {item.b} ="
    forced_w = f"{item.a} + {item.b} ={item.W_str}.{suffix}"
    self_gen = f"{gen.full_prefix}{suffix}"
    # Reconstruct with canonical spacing from token join
    forced_match = f"{gen.prefix}{gen.impulse_str}{suffix}"
    return {
        "forced_W": forced_w,
        "self_generated": self_gen,
        "forced_impulse_match": forced_match,
    }


def evaluate(model, item, mode: str, prompt: str, gen: GenResult) -> dict:
    str_toks = model.to_str_tokens(prompt)
    t_star = find_t_star(str_toks)
    W_id = model.to_single_token(item.W_str)
    C_id = model.to_single_token(item.C_str)
    logits = model(model.to_tokens(prompt))
    v = logits[0, t_star]
    score = float(v[W_id] - v[C_id])
    top1 = model.to_string([int(v.argmax())])
    p_w, p_c = float(v[W_id]), float(v[C_id])
    p_c_pair = p_c / (p_w + p_c)

    gen_first = gen.first_token
    gen_top1_match = top1 == gen_first if gen_first else False

    return {
        "item": item.name,
        "mode": mode,
        "prompt": prompt,
        "t_star": t_star,
        "score_bank_WC": score,
        "p_C_pair": p_c_pair,
        "top1": top1,
        "top1_is_bank_W": top1 == item.W_str,
        "top1_is_bank_C": top1 == item.C_str,
        "top1_is_gen_first": gen_top1_match,
        "gen_impulse_str": gen.impulse_str,
        "gen_first_token": gen_first,
        "gen_equals_bank_W": gen_first == item.W_str,
        "gen_equals_bank_C": gen_first == item.C_str,
    }


def plot_results(df: pd.DataFrame):
    modes = ["forced_W", "self_generated", "forced_impulse_match"]
    colors = {"forced_W": "#3498db", "self_generated": "#e74c3c", "forced_impulse_match": "#95a5a6"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    summary = df.groupby("mode").agg(
        mean_score=("score_bank_WC", "mean"),
        sem_score=("score_bank_WC", "sem"),
        frac_top1_bank_W=("top1_is_bank_W", "mean"),
        frac_top1_gen=("top1_is_gen_first", "mean"),
    ).reindex(modes)

    x = np.arange(len(modes))
    axes[0].bar(
        x,
        summary["mean_score"],
        yerr=summary["sem_score"],
        capsize=4,
        color=[colors[m] for m in modes],
        alpha=0.9,
    )
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=15, ha="right")
    axes[0].set_ylabel("bank score (logit W − logit C) @ t*")
    axes[0].set_title("E8 — forced vs self-generated impulse")

    width = 0.35
    axes[1].bar(x - width / 2, summary["frac_top1_bank_W"], width, label="top-1 = bank W", color="#3498db")
    axes[1].bar(x + width / 2, summary["frac_top1_gen"], width, label="top-1 = gen 1st tok", color="#e74c3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, rotation=15, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Behavioral stickiness")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "e8_forced_vs_selfgen.png", dpi=150)
    plt.close(fig)

    # per-item paired plot
    fig, ax = plt.subplots(figsize=(8, 4))
    items = df["item"].unique()
    forced = df[df["mode"] == "forced_W"].set_index("item")["score_bank_WC"]
    selfg = df[df["mode"] == "self_generated"].set_index("item")["score_bank_WC"]
    xi = np.arange(len(items))
    ax.bar(xi - 0.2, [forced[i] for i in items], width=0.4, label="forced_W", color=colors["forced_W"])
    ax.bar(xi + 0.2, [selfg[i] for i in items], width=0.4, label="self_generated", color=colors["self_generated"])
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(xi)
    ax.set_xticklabels(items, rotation=30, ha="right")
    ax.set_ylabel("bank W−C score")
    ax.legend()
    ax.set_title("E8 per-item paired scores")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e8_per_item.png", dpi=150)
    plt.close(fig)


def main():
    print("device:", DEVICE)
    model = tl.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    rows = []
    gen_by_item = {}
    for item in ITEMS:
        prefix = build_prefix(item)
        gen = greedy_generate_impulse(model, prefix)
        gen_by_item[item.name] = gen
        prompts = build_prompts(item, gen)
        for mode, prompt in prompts.items():
            print(item.name, mode, "impulse:", repr(gen.impulse_str[:30]))
            rows.append(evaluate(model, item, mode, prompt, gen))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "forced_vs_selfgen.csv", index=False)

    forced = df[df["mode"] == "forced_W"].set_index("item")
    selfg = df[df["mode"] == "self_generated"].set_index("item")
    match = df[df["mode"] == "forced_impulse_match"].set_index("item")

    delta_self = selfg["score_bank_WC"] - forced["score_bank_WC"]
    delta_match = match["score_bank_WC"] - selfg["score_bank_WC"]

    summary = pd.DataFrame({
        "mean_score_forced_W": [forced["score_bank_WC"].mean()],
        "mean_score_self_generated": [selfg["score_bank_WC"].mean()],
        "mean_score_forced_match": [match["score_bank_WC"].mean()],
        "mean_delta_self_minus_forced": [delta_self.mean()],
        "sem_delta_self_minus_forced": [delta_self.std() / np.sqrt(len(delta_self))],
        "mean_delta_match_minus_self": [delta_match.mean()],
        "frac_top1_bank_W_forced": [forced["top1_is_bank_W"].mean()],
        "frac_top1_bank_W_self": [selfg["top1_is_bank_W"].mean()],
        "frac_top1_gen_first_self": [selfg["top1_is_gen_first"].mean()],
        "frac_gen_first_equals_bank_W": [selfg["gen_equals_bank_W"].mean()],
        "frac_gen_first_equals_bank_C": [selfg["gen_equals_bank_C"].mean()],
    })
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.T)

    per_item = pd.DataFrame({
        "gen_first_token": selfg["gen_first_token"],
        "gen_impulse": selfg["gen_impulse_str"],
        "score_forced_W": forced["score_bank_WC"],
        "score_self": selfg["score_bank_WC"],
        "score_match": match["score_bank_WC"],
        "delta_self_forced": delta_self,
        "top1_forced": forced["top1"],
        "top1_self": selfg["top1"],
    })
    per_item.to_csv(OUT_DIR / "per_item.csv")

    plot_results(df)

    mean_forced = float(forced["score_bank_WC"].mean())
    mean_self = float(selfg["score_bank_WC"].mean())
    mean_match = float(match["score_bank_WC"].mean())
    verdict = {
        "persistence_only_when_W_typed": bool(
            mean_forced > mean_self + 1.0 and forced["top1_is_bank_W"].mean() >= 0.99
        ),
        "typing_same_as_generating_impulse": bool(abs(mean_match - mean_self) < 0.15),
        "self_gen_sticks_to_own_impulse": bool(selfg["top1_is_gen_first"].mean() >= 0.75),
        "mean_score_forced_W": mean_forced,
        "mean_score_self_generated": mean_self,
        "mean_score_forced_impulse_match": mean_match,
        "mean_delta_self_minus_forced": float(delta_self.mean()),
        "frac_top1_bank_W_forced": float(forced["top1_is_bank_W"].mean()),
        "frac_top1_bank_W_self": float(selfg["top1_is_bank_W"].mean()),
        "interpretation": (
            "bank_W_persistence_is_input_copying"
            if mean_forced > mean_self + 1.0
            else "persistence_survives_self_generation"
        ),
    }
    with open(OUT_DIR / "e8_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("verdict:", json.dumps(verdict, indent=2))
    print("Wrote outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

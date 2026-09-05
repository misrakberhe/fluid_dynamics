# Executive summary — MATS application

**Paste-ready for the form.** ~350 words.  
**Full write-up:** [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md) · **Figure:** [`figures/flagship_ABC.png`](figures/flagship_ABC.png) · **Repo:** https://github.com/misrakberhe/fluid_dynamics

---

## Claim

**Claim:** On Qwen3.5-4B, the model can *say* the revised (correct) answer while still having an editable internal copy of the typed wrong answer.

- **Behavior:** after “Wait, let me recompute,” greedy output is C on every item in an 8-item set.
- **Causal check:** overwriting residual activations in the **middle layers**, at the token positions around that typed answer, still shifts preference (Δ ≈ −2.1 vs ≈ 0 at random positions; bootstrap 95% CI [−2.43, −1.69]).

So revision changed the output without removing that pathway. Surface correction is not evidence that the mechanism is gone.

## Setup

Template: `{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =`. Score = logit(W) − logit(C) at the final `=` (or Qwen `answer_pos`); also greedy top-1. GPT-2 small maps the mechanism (8 multi-digit items). Qwen3.5-4B is the external check (8 single-digit items; multi-digit answers split under Qwen tokenization).

## GPT-2 mechanism (supporting map)

Revision cues never flip greedy output to C (top-1 = typed W on 8/8; mean score ≈ +2.71). Residual patching shows mid-depth state at the typed-answer tokens is necessary and sufficient for later preference (C-swap Δ ≈ −4.9; W-patch Δ ≈ +4.9; largest effect L5–L8). Blocking new writes there barely moves the score (Δ ≈ −0.3)—accumulated residual matters, not fresh writes. Scope: on this item set, persistence of the designated wrong answer requires that answer in the prompt (not generic self-commitment).

## Qwen dissociation (main result)

| | GPT-2 | Qwen3.5-4B |
|---|---|---|
| Forced-W revision successful? | No | Yes (8/8 → C) |
| C-swap at typed-answer tokens | Δ ≈ −4.9 | Δ ≈ −2.1 vs rand ≈ 0 |

All eight Qwen C-swap Δs are negative. Effect is smaller than GPT-2 but clearly above the off-locus control.

## Limitations and next step

n = 8 per model; GPT-2 is a minimal arithmetic testbed; we do not claim this for all post-trained models or isolate RL vs instruction tuning. Next: enlarge the Qwen set and/or sweep layers / prompt framing to see when behavioral sticking and causal patchability travel together vs come apart.

**Artifacts:** full write-up, flagship figure, and runnable E4 / Qwen scripts in the repo above.

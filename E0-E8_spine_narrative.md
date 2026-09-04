# E0–E8 Spine Narrative

**Project:** forced-error arithmetic on GPT-2 small  
**Prompt skeleton:** `{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =`  
**Probe:** final `=` (`t*`); score = logit(W) − logit(C)  
**Date frozen:** 2026-08-31

---

## The question

Does something get **committed** at a locatable point in the forward pass and **carry forward** to
influence a later answer — even when the model is asked to reconsider? Or is “sticking” something
else (input-copying, routing habit, surface behavior)?

---

## What we measured (in order)

| Phase | Exp | One-line result |
|---|---|---|
| Methods | **E0** | Tuned lens is the primary meter; logit lens is prompt-dependent |
| Baseline | **E1** | Cues **never** flip greedy answer to C; W sticks everywhere |
| Spine | **E2** | W preference is **written** post-embed as a **late write-wave** (attn + MLP), not one mid-depth kick |
| Spine | **E3** | At `t*`, preference **lands** mid-depth (L5–L11); OV routing from **W_window** dominates |
| Spine | **E4** | **W_window residual is causally necessary/sufficient** (Δ ≈ ±5); leverage at **L5–L8** |
| Spine | **E4b** | Readout at `t*` emerges ~L4; **causal** fixation ~L10 — readable ≠ necessary |
| Spine | **E5** | Routing `t*` → W_window **matters** but weaker than content (Δ ≈ −2.3 vs −4.9) |
| Spine | **E6** | Operand corruption **softens** W preference (Δ ≈ −2.3) but model does **not** recompute visible sum |
| Spine | **E7** | Revision cue does **not** write toward C — “momentum resists cue” is wrong frame |
| Gate | **E8** | Bank-W persistence **only when W is typed in**; self-gen sticks to **own** greedy token |

---

## The mechanistic picture (forced-error regime)

When **W is in the prompt** as typed input:

1. **Write:** Late layers add a distributed write-wave along the W−C direction around the forced-answer
   locus — not a single sharp impulse, not mostly embedding.
2. **Commit (causal):** The residual state at **W_window** mid-depth (especially L5–L8) **determines**
   whether `t*` prefers W or C. Patching/swapping that state moves the answer.
3. **Read:** Late layers at `t*` (L9–L11) are causally load-bearing for the final preference, even
   though an earlier readout can already weakly favor W.
4. **Route:** Attention from `t*` to W_window helps, but blocking routes hurts less than swapping
   content — backup paths exist.
5. **Modulate:** Visible second-instance operands and revision-span text **modulate** sticking; they
   do not cleanly override it or recruit correction toward C.

**Behavioral floor:** revision language does not produce C at `t*` (E1). Internally, the cue is not
recruited as a competing overwrite toward C (E7).

---

## What E8 changed

E8 compared **typed bank W** vs **greedy self-generated** answers before the cue.

| Condition | Mean W−C @ `t*` | Top-1 |
|---|---|---|
| Forced W in prompt | +2.71 | bank W (100%) |
| Self-generated impulse | +0.57 | model’s own token (100%) |
| Typed same string as self-gen | +0.55 | same as self-gen |

- The forced/self gap is **which token** sits in the impulse slot, not typing vs generating the
  same substring (Δ ≈ 0 between self-gen and matched forced).
- GPT-2 does **not** greedily emit bank W/C from `{a} + {b} =` on this 8-item bank (0/8).

**Scope correction:** E2–E7 describe a real, causal pipeline for **carrying a typed-in wrong answer
forward**. They do **not** license a general claim that the model independently commits to the same
wrong arithmetic answer.

### Forced W vs forced C (post-spine)

| Condition | Mean W−C @ `t*` | Top-1 = impulse |
|---|---|---|
| Forced W in prompt | +2.71 | 100% (bank W) |
| Forced C in prompt | −2.68 | 100% (bank C) |

Asymmetry ratio |W|/|C| ≈ 1.01; mirrored causal swaps Δ ≈ ±5. **Wrong and correct anchoring are
symmetric** — persistence is generic answer-slot binding, not wrong-answer bias. See
`forced_W_vs_C_session_summary.md`.

### Self-aligned persistence (post-spine)

E8 scored self-gen on bank W−C (+0.57) vs forced W (+2.71) — misleading when gen ≠ bank W.
Re-scoring on each item's axis (logit(gen) − logit(bank C)): self-gen **+7.46**, 100% top-1 = gen;
causal gen-locus swap Δ ≈ −10. **Generic slot binding** for self-generated impulses. See
`self_aligned_persistence_session_summary.md`.

---

## What we can claim

**Supported (forced-error regime, GPT-2 small, this prompt family):**

- A locatable residual state at the forced-answer locus is **causally necessary** for W preference at `t*`.
- That state is built by a late write-wave, integrated mid-depth, read/causally fixed late at `t*`.
- Sticking is **modulated** by visible context but not explained by pure recomputation or cue-driven
  overwrite toward C.
- Revision cues **fail behaviorally** and are **not** recruited internally toward C.
- Forced **correct** (C) anchoring is **symmetric** to forced wrong (W) on this bank — not a wrong-answer bias.
- Self-generated impulses **persist strongly** on each item's aligned axis (+7.5, 100% top-1 = gen); E8's low bank W−C score was ruler mismatch.

**Not supported without redesign:**

- Generic “commitment” or “momentum” independent of having W in the input.
- “The model resists revision” — nothing competes; the cue doesn’t write toward C.
- Generalization to self-generated wrong answers on this bank (model emits different tokens).

**External validity / headline extension (Qwen3.5-4B):** On an 8-item single-digit bank, forced-W revision
**succeeds** behaviorally (0% top-1 = W; mean score −3.38 ± SEM 0.26 @ answer_pos) unlike GPT-2 (+2.7, 100% W).
Yet W_window C-swap at L8–23 still shifts preference (Δ −2.10, SEM 0.21; bootstrap 95% CI [−2.43, −1.69];
all 8 items negative; random-position control ≈ 0) — ~43% of GPT-2's magnitude (−4.9). Mid-depth
answer-locus state remains a causal lever even when greedy output already favors C: **revision can
mask rather than remove an anchoring mechanism.** Blog: [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md).
Details: [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md).

---

## One-sentence summary

We found a **mechanism for persisting a typed-in mistake** through depth (write → mid-depth state →
late readout), not evidence that GPT-2 **self-commits** to the same wrong answer when it generates
its own.

---

## Open forks

- **Write-up:** frame E2–E7 as forced-error mech interp; E8 as scope boundary.
- **Done (post-spine):** forced W vs C asymmetry — near-symmetric; see `forced_W_vs_C_session_summary.md`.
- **Done (post-spine):** self-aligned persistence — gen sticks strongly on aligned axis; see `self_aligned_persistence_session_summary.md`.
- **Done (post-spine):** Qwen3.5-4B replication — behavioral revision succeeds; causal C-swap still localizes; see `qwen_anchoring_replication_session_summary.md`.
- **Redesign:** prompt bank where greedy gen hits bank W; re-run E8.
- **Phase 4 (optional, label forced-only):** E9 formation threshold, E10 re-excitation, E11 QK resonance.
- **Deferred:** J-lens; E12 transfer function.

**Artifacts:** `E{N}_session_summary.md`, `E{N}_*.py`, `E{N}_outputs/` for N = 0–8 (+ E4b);
post-spine extensions use descriptive names (e.g. `forced_W_vs_C_*`).

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

---

## What we can claim

**Supported (forced-error regime, GPT-2 small, this prompt family):**

- A locatable residual state at the forced-answer locus is **causally necessary** for W preference at `t*`.
- That state is built by a late write-wave, integrated mid-depth, read/causally fixed late at `t*`.
- Sticking is **modulated** by visible context but not explained by pure recomputation or cue-driven
  overwrite toward C.
- Revision cues **fail behaviorally** and are **not** recruited internally toward C.

**Not supported without redesign:**

- Generic “commitment” or “momentum” independent of having W in the input.
- “The model resists revision” — nothing competes; the cue doesn’t write toward C.
- Generalization to self-generated wrong answers on this bank (model emits different tokens).

---

## One-sentence summary

We found a **mechanism for persisting a typed-in mistake** through depth (write → mid-depth state →
late readout), not evidence that GPT-2 **self-commits** to the same wrong answer when it generates
its own.

---

## Open forks

- **Write-up:** frame E2–E7 as forced-error mech interp; E8 as scope boundary.
- **Redesign:** prompt bank where greedy gen hits bank W; re-run E8.
- **Phase 4 (optional, label forced-only):** E9 formation threshold, E10 re-excitation, E11 QK resonance.
- **Deferred:** J-lens; E12 transfer function.

**Artifacts:** `E{N}_session_summary.md`, `E{N}_*.py`, `E{N}_outputs/` for N = 0–8 (+ E4b).

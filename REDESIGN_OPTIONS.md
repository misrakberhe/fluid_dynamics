# Redesign Options — After E0–E8

**Context (2026-08-31):** The spine (E0–E8) is complete. E2–E7 map a causal pipeline for
**typed-in wrong-answer persistence** in the forced-error regime. E8 shows bank-W sticking
requires W in the prompt on the current 8-item bank (GPT-2 small); self-gen sticks to its
own greedy token instead.

This file lists **forward options** — not a commitment to any path. See
[`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) for what was established.

---

## Priority sketch (if choosing one path)

| Priority | Option | Why |
|---|---|---|
| **A** | [1 — Own & generalize forced anchoring](#1-own--generalize-the-narrower-result-forced-answer-anchoring) | Complete causal story already in hand — **draft:** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) |
| **B** | [3 — Self-aligned persistence (E8b)](#3-self-aligned-persistence-e8b) | Cheap; tests generic answer-slot binding |
| **C** | [2 — Bank/model redesign for E8](#2-redesign-bank-or-model-so-self-gen-matches-forced) | Revives strong “commitment” claim if it passes |
| **D** | [4 — Domain where revision sometimes works](#4-domain-where-revision-sometimes-works) | Revives original E1 momentum / revision question |

---

## 1. Own & generalize the narrower result (forced answer anchoring)

**What you have:** E4-level finding — residual at the answer locus (W_window) **causally**
determines preference at `t*`; write-wave (E2), routing (E3/E5), readout ≠ causality (E4b);
cue does not write toward C (E7); operand corruption modulates but does not fully reset (E6).

**Reframe as:** *How do transformers persist a token presented in context as a provisional
answer, and why don’t revision cues override it?*

**Generalize by:**
- More models (not only GPT-2 small)
- More prompt shapes (cue phrasing, operand formats, W vs C forcing)
- Compare to adjacent phenomena (ICL priming, user-planted facts, sycophancy)
- Optional Phase 4 within forced regime: E9 formation threshold, E10 re-excitation

**Nontrivial if:** positioned as mechanistic **answer-slot / context anchoring**, with causal
interventions — not as generic “momentum.”

**Effort:** Low–medium (mostly write-up + targeted extensions).

---

## 2. Redesign bank or model so self-gen matches forced

**Problem:** On the current bank, greedy gen from `{a} + {b} =` never emits bank W (0/8). E8
compared typed `25` vs typed `1`/`8`/… — not forced vs self-generated **same** impulse.

**Redesign options:**
- **Search bank:** find `a+b=` where greedy top-1 equals chosen W (wrong-sum token)
- **Partial prefix:** steer generation (e.g. `12 + 15 = 2` → `5`) — fragile
- **Better model:** arithmetic-capable LM that emits sum-like tokens
- **Non-arithmetic domain:** reliable wrong completions on GPT-2 or small models

**Re-run:** E8 (forced W vs self-generated same W); optionally E4 patching at that locus.

**Nontrivial if:** self-gen ≈ forced on bank W−C score **and** E4-style residual swap still
moves `t*`. Legitimizes “commitment” for model-chosen impulses, not only typed ones.

**Effort:** Medium (bank search + re-run E8 ± E4).

---

## 3. Self-aligned persistence (E8b)

**Problem:** Scoring self-gen on bank W−C asks the wrong question when gen ≠ bank W.

**Protocol:**
- Impulse = `gen_token` (first token after first `=`)
- Score = logit(gen) − logit(contrast), e.g. bank C or runner-up at gen time
- Compare strength to forced-W on **each item’s own axis**
- Optional: E4-style residual patch at **gen** position (not W_window for bank W)

**Question:** Is persistence **generic** (“stick to whatever occupied the answer slot”) with
similar causal strength for gen vs forced?

**Nontrivial if:** gen persistence is as strong as forced-W on matched rulers and causally
localized at the impulse locus.

**Effort:** Low (reuse E8 + E4 machinery).

---

## 4. Domain where revision sometimes works

**Problem:** E1 revision curve is flat — 0% top-1 = C. Original Group 1 question needs
settings where the cue **occasionally** redirects behavior.

**Redesign:**
- Capable models + multi-step math, logic, or code
- Cues that sometimes work in the literature (“wait”, “actually”, …) on reasoning traces
- Cue placement **after model-generated** chains, not only forced wrong tokens

**Then:** mechanistic work explains **success vs failure** (patching/routing predicts revision?).

**Nontrivial if:** non-flat revision curve — structural measures must beat token index alone.

**Effort:** High (new domain, new baselines, possibly new tooling).

---

## 5. Force C, or wrong vs right asymmetry

**Untested symmetry:** spine mostly used **forced wrong** (bank W).

**Variants:**
- **Forced C:** `12 + 15 = 27. Wait…` — does correct answer persist equally?
- **Asymmetry:** is wrong-W anchoring stronger than correct-C anchoring?

**Nontrivial if:** wrong and right anchoring differ — practically and psychologically interesting.

**Effort:** Low (reuse full spine with C-prompt as primary; E4 already has `build_C_prompt`).

---

## 6. Real model errors (not planted mistakes)

**Current setup:** mistake is **planted** in the prompt.

**Protocol:**
1. Prompt until model **naturally** produces a wrong completion (or sample until wrong).
2. Freeze trajectory.
3. Append revision cue **without** retyping the wrong answer.

**Nontrivial if:** natural errors show the same E4 causal structure as forced W.

**Effort:** Medium–high (generation harness + item selection).

---

## 7. Richer impulse than one token

GPT-2 greedy continuations are often odd (` 1.`, ` 8.`). Commitment may live in:
- Multi-token rationales
- Wrong **derivation** spans before the cue (cf. E1 Design B)
- Chain-of-thought steps on capable models

**Redesign:** impulse = **span**; patch/ablate span, not single token.

**Nontrivial if:** persistence scales with wrong **structure** in context, not one embed lookup.

**Effort:** Medium–high.

---

## 8. Cross-model / scale

GPT-2 small on `a+b=` is a weak domain (weird gens, no revision success).

**Same protocol on** larger LMs may yield:
- Self-generated wrong answers aligned with task
- Occasional cue success
- Stronger routing/write structure

**Nontrivial if:** E8 gate **passes** on a capable model — replicate E2–E7 there.

**Effort:** High (compute, TransformerLens / NNsight compatibility).

---

## 9. Comparative framing (literature positioning)

Rhymes with:
- Answer anchoring / sycophancy (user-supplied wrong facts)
- ICL exemplar binding
- Induction heads / previous-token copying
- “Wait” self-correction on reasoning models (different regime)

**Redesign as comparison:** same E4 patch on user-planted vs ICL-planted vs self-generated
wrong answers.

**Effort:** Medium (framing + selective experiments).

---

## What E0–E8 ruled out (don’t reclaim without redesign)

- Generic “momentum resists revision” (E1, E7)
- Pure carried residue vs pure recompute (E6 — mixed)
- Bank-W persistence without W in input on current bank (E8)
- General self-commitment to wrong arithmetic on this bank (E8)

---

## Related artifacts

- [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) — one-page spine summary
- [`momentum-experiment-plan.md`](momentum-experiment-plan.md) — original plan (header stale)
- [`NOTES.md`](NOTES.md) — research notes + pointer to spine

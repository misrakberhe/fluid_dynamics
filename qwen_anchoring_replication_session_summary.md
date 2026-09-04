# Qwen Anchoring Replication — Session Summary

**Date:** 2026-08-31  
**Model:** `Qwen/Qwen3.5-4B` (32 layers, CUDA)  
**Artifacts:** `qwen_anchoring_replication.py`, `qwen_replication_outputs/`  
**Plan reference:** [`plan_qwen_anchoring_replication.md`](plan_qwen_anchoring_replication.md)

**Naming note:** Post-spine external-validity extension. Not numbered E9+ — tests whether GPT-2 forced-answer anchoring (behavior + E4-style patching) holds on a modern dense LM.

---

## 1. Working setup

**Item bank:** 8 single-digit addition items (`QWEN_ITEMS`) — GPT-2's multi-digit bank fails Qwen tokenization (digits split per character).

| Variant | Skeleton |
|---|---|
| **forced W** | `{a} + {b} ={W}. Wait, let me recompute. {a} + {b} =` |
| **forced C** | `{a} + {b} ={C}. Wait, let me recompute. {a} + {b} =` |

**Scoring (Qwen-specific):** At the second `=`, Qwen's greedy continuation is whitespace before the answer digit. Primary metrics use **`answer_pos`** (position after greedy space prefix), not raw `t*`. GPT-2 reference numbers are scored at `t*` directly (digit is top-1 there).

**Causal mirror:** impulse-window `resid_post` C-swap on W-prompt, layers **L8–23** (25–75% depth on 32 layers). Random-position control matched window width.

**Loader:** TransformerLens 3 `TransformerBridge.boot_transformers(...)`.

---

## 2. Gate results

| Gate | Criterion | Result |
|---|---|---|
| **G1 — Behavior** | forced-W: mean score > +1.0 and ≥6/8 top-1 = W | **FAIL** |
| **G2 — Causal** | W_window C-swap mean Δ < −1.0 | **PASS** |
| **G3 — Honest report** | Document failures | **PASS** |

Source: `qwen_replication_outputs/verdict.json`

---

## 3. Empirical conclusions

### Behavior: revision succeeds (unlike GPT-2)

| Condition | GPT-2 @ t* | Qwen @ answer_pos |
|---|---|---|
| **forced W** mean W−C | **+2.71** | **−3.38** |
| **forced W** frac top-1 = impulse | **1.00** | **0.00** |
| **forced W** frac top-1 = correct (C) | 0% | **100%** |
| **forced C** mean W−C | −2.68 | −5.50 |
| **forced C** frac top-1 = impulse | 1.00 | 1.00 |

**Read:** On forced-W prompts, Qwen **revises to the correct answer** at the second `=` — no GPT-2-style sticking to the typed wrong digit. Forced-C still shows impulse persistence (100% top-1 = C), so the asymmetry is **wrong-specific revision success**, not generic failure to read the slot.

### Causal: mid-depth C-swap still shifts preference

| Intervention | mean Δ (W−C score) | SEM | bootstrap 95% CI | GPT-2 ref |
|---|---|---|---|---|
| **W_window C-swap** (L8–23) | **−2.10** | 0.21 | **[−2.43, −1.69]** | −4.94 (L5–11) |
| **Random-position control** | **−0.02** | 0.04 | — | (not run in ref) |

**Per-item C-swap Δ** (all negative): −2.76, −2.49, −2.32, −2.31, −2.29, −2.01, −1.76, −0.86.  
Seven of eight items in [−2.76, −1.76]; one weaker item still negative — mean not outlier-driven.

**Read:** Residual state at the answer locus remains a **localized causal lever** on Qwen — C-swap pushes preference toward C (more negative W−C score) while off-locus swap is null. Effect is **~43% of GPT-2 magnitude** but clearly above control (CI excludes 0).

### Behavior–causality dissociation

GPT-2 shows **both** behavioral sticking and strong causal patching. Qwen shows **revision success behaviorally** but **non-null causal localization** — the mid-depth state still matters for answer-digit preference even when greedy output already favors C.

This is the most interesting outcome for external validity: anchoring is not a monolithic “capable models revise” vs “GPT-2 sticks” story; **causal structure can survive behavioral revision**.

---

## 4. Protocol notes

1. **Token audit:** GPT-2 item bank scored 0/8 pass on Qwen; dedicated single-digit bank required.
2. **Whitespace at t*:** Qwen predicts space before digit; `answer_pos` fix is mandatory for fair W vs C comparison (see `verdict.json` note).
3. **Layer band:** L8–23 chosen by 25–75% depth rule; no per-layer sweep yet (stretch G5).

---

## 5. Comparison to GPT-2 spine

| Dimension | GPT-2 small | Qwen3.5-4B |
|---|---|---|
| Behavioral anchoring (forced W) | Strong (+2.7, 100% W) | Absent (revises to C) |
| Forced C persistence | Symmetric | C still sticks |
| C-swap causal Δ | −4.94 | −2.10 |
| Rand control | — | ≈ 0 |
| Revision cue effect | Fails (0% flip) | Succeeds on forced W |

---

## 6. What this does *not* establish

- Full replication of GPT-2 anchoring on a second model (G1 failed).
- Whether a narrower layer band or different prompt framing restores behavioral sticking.
- Generalization beyond this 8-item single-digit bank.
- Mechanistic identity of the Qwen write/read pathway (no E2–E4b spine on Qwen).

---

## 7. Figures & files

| File | Description |
|---|---|
| `behavior_gpt2_vs_qwen.png` | Behavior bars: mean score + top-1 fractions |
| `causal_interventions.png` | Qwen C-swap vs rand control |
| `replication_causal_gpt2_vs_qwen.png` | Side-by-side causal Δ (Phase 4) |
| `behavior.csv`, `causal.csv` | Per-item records |
| `verdict.json` | Gate summary + GPT-2 refs |

**Reproduce:** `python qwen_anchoring_replication.py phase4` (figures from saved CSVs; no GPU).

---

## 8. Short takeaway

Qwen3.5-4B **does not** behaviorally anchor forced wrong answers — revision cues work on this bank. Mid-depth impulse-window C-swap **still** shifts answer-digit preference (Δ ≈ −2.1 vs ≈ 0 control), at weaker magnitude than GPT-2. External validity is **partial**: causal localization generalizes; behavioral sticking does not.

**MATS one-liner:**

> Qwen3.5-4B does not behaviorally anchor forced wrong answers (revision succeeds), but mid-depth impulse-window C-swap still shifts answer-digit preference (Δ ≈ −2.1 vs ≈ 0 control) — causal localization without GPT-2-style sticking.

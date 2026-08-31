# Forced W vs C Session Summary — Wrong vs right anchoring asymmetry

**Date:** 2026-08-31  
**Artifacts:** `forced_W_vs_C_asymmetry.py`, `forced_W_vs_C_outputs/`  
**Plan reference:** `REDESIGN_OPTIONS.md` §5 (Force C / wrong vs right asymmetry)

**Naming note:** Not numbered E9 — `E9` in `momentum-experiment-plan.md` is reserved for
formation-threshold patching. This is a post-spine extension testing whether anchoring is
wrong-specific or generic answer-slot binding.

---

## 1. Working setup

Two prompt variants per item (8-item bank, GPT-2 small):

| Variant | Skeleton |
|---|---|
| **forced W** | `{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =` (E1–E8 baseline) |
| **forced C** | `{a} + {b} = {C}. Wait, let me recompute. {a} + {b} =` |

**Score at \(t^*\):** \(\mathrm{logit}(W) - \mathrm{logit}(C)\) (same bank contrast pair).

**Causal mirror (L5–L11):** impulse-window `resid_post` swap between runs — W-prompt gets
C-run state; C-prompt gets W-run state. (Same machinery as E4 necessity / sufficiency.)

---

## 2. Empirical conclusions

### Anchoring is **near-symmetric** for wrong vs correct

| Condition | mean score @ \(t^*\) | frac top-1 = impulse | causal Δ (swap) |
|---|---|---|---|
| **forced W** | **+2.71** | **1.00** | **−4.94** |
| **forced C** | **−2.68** | **1.00** | **+4.92** |

**Asymmetry ratio** |W| / |C| ≈ **1.01** (behavioral and causal).

**Read:** whatever token occupies the answer slot — wrong or correct — persists equally at
\(t^*\). The revision cue fails in both directions (0% flip to the alternate answer). Causal
flip magnitude is also symmetric. Persistence is **generic answer-slot binding**, not a bias
toward wrong answers.

### Relation to prior work

- E4 already embedded C-prompt baselines and sufficiency; this experiment packages the asymmetry
  question explicitly.
- E8 showed bank-W sticking requires W in the prompt; forced C shows the mirror holds for C.

---

## 3. What this does *not* establish

- Asymmetry on other models, domains, or cue phrasings.
- Whether self-generated impulses (E8) show the same symmetry when scored on matched axes.

---

## 4. Short takeaway

Forced wrong and forced correct anchoring are **symmetric** on this bank: mean scores ±2.7,
100% top-1 = impulse token, and mirrored residual swaps flip preference by Δ ≈ ±5. Wrong-answer
persistence is not special — it is the same mechanism as persisting a typed-in correct answer.

# Self-aligned Persistence Session Summary

**Date:** 2026-08-31  
**Artifacts:** `self_aligned_persistence.py`, `self_aligned_persistence_outputs/`  
**Plan reference:** `REDESIGN_OPTIONS.md` §3 (Self-aligned persistence)

**Naming note:** Post-spine extension; not numbered E8b in filenames. Follows E8 but scores each
item on its **own axis** (gen token vs bank C), not bank W−C.

---

## 1. Working setup

Per item (8-item bank, GPT-2 small):

| Condition | Impulse in slot | Score ruler @ \(t^*\) |
|---|---|---|
| **self_generated** | Greedy gen from `{a}+{b}=` | logit(gen) − logit(bank C) |
| **forced_gen** | Same gen token typed in | logit(gen) − logit(bank C) |
| **forced_bank_W** (reference) | Bank W typed in | logit(W) − logit(C) |

**Causal (L5–L11):** on forced-gen prompt, swap impulse-window `resid_post` with bank-C-run state;
measure Δ on aligned axis.

---

## 2. Empirical conclusions

### E8's weak self-gen score was a **ruler mismatch**

| Condition | mean score @ \(t^*\) | frac top-1 = impulse |
|---|---|---|
| **self_generated** (aligned) | **+7.46** | **1.00** (gen token) |
| **forced_gen** (aligned) | **+7.46** | **1.00** |
| **forced_bank_W** (bank W−C) | **+2.71** | **1.00** (bank W) |

E8 reported self-gen mean **+0.57** on bank W−C — because gen ≠ bank W on 8/8 items. On the
**correct ruler**, self-gen persistence is **strong** (100% top-1 = gen), not weak.

### Typing vs generating: **no difference** (replicates E8)

mean aligned: self_gen = forced_gen; Δ = **0.00**.

### Causal localization at **gen locus**

Impulse-window C-swap: mean Δ = **−10.11** on aligned axis (vs **−4.94** for bank-W C-swap in
`forced_W_vs_C`). Gen persistence is causally localized; flip magnitude is ~2× bank-W (likely
larger gen−C logit separation than W−C).

**Read:** persistence is **generic answer-slot binding** — stick to whatever occupies the slot,
measured on that token's axis. E8's "input-copying" framing for bank W is correct; self-gen
**does** persist strongly when scored fairly.

---

## 3. Relation to E8 and forced W vs C

- **E8:** bank-W sticking requires W in prompt; self-gen sticks to own token, not bank W.
- **Self-aligned:** on gen axis, self-gen ≈ forced-gen and **stronger** than forced bank-W on
  bank axis (+7.5 vs +2.7).
- **forced_W_vs_C:** wrong and correct typed anchoring are symmetric; self-aligned shows gen
  anchoring is also strong when the ruler matches.

---

## 4. What this does *not* establish

- Whether gen persistence equals forced-gen when gen happens to equal bank W (0/8 here).
- Runner-up contrast (used bank C only).
- Generality beyond this bank/model.

---

## 5. Short takeaway

Scoring self-gen on bank W−C understates persistence. On each item's aligned axis (gen vs bank C),
self-generated impulses persist at **+7.5** with 100% top-1 = gen — stronger than forced bank-W
on its axis (+2.7). Typing vs generating is identical. Causal swap at gen locus flips preference
(Δ ≈ −10). **Generic slot binding** holds for self-generated answers, not only typed wrong ones.

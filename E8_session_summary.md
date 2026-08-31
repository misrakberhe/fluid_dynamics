# E8 Session Summary — Self-generated vs forced impulse

**Date:** 2026-08-31  
**Artifacts:** `E8_self_generated_impulse.py`, `E8_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 3 (E8)

Phase 3 gate: is bank-`W` persistence at \(t^*\) **commitment**, or **input-copying** of a
forced wrong-answer token in the prompt?

---

## 1. Working setup

Three conditions per item (8-item bank, GPT-2 small):

| Condition | First-instance answer |
|---|---|
| **forced_W** | Bank `W` typed in prompt: `{a}+{b}={W}.` + cue + suffix (E1–E7 baseline) |
| **self_generated** | Greedy generate from `{a}+{b}=` until `.` or `\n` (max 4 tokens); freeze; append cue |
| **forced_impulse_match** | Teacher-force the **exact** self-generated impulse string; append cue |

**Score at \(t^*\):** bank \(\mathrm{logit}(W) - \mathrm{logit}(C)\) (same W/C as forced-error bank).

**Note:** GPT-2 rarely generates bank `W` or `C` from `{a}+{b}=` (greedy first tokens are
` 1`, ` 8`, ` 4`, etc.). E8 therefore tests whether **bank-W sticking** requires `W` in the
input, not whether the model self-commits to the same wrong answer it was never asked to produce.

---

## 2. Empirical conclusions

### Bank-W persistence is **forced-input specific**

| Condition | mean bank W−C @ \(t^*\) | frac top-1 = bank `W` | frac top-1 = gen 1st tok |
|---|---|---|---|
| forced_W | **+2.71** | **1.00** | — |
| self_generated | **+0.57** | **0.00** | **1.00** |
| forced_impulse_match | **+0.55** | 0.00 | 1.00 |

**Δ (self − forced):** mean **−2.13** (SEM ≈ 0.41).

**Read:** sticking on bank `W` at \(t^*\) appears **only when `W` is typed into the prompt**.
Self-generated impulses do **not** reproduce bank-W preference.

### Self-gen sticks to **its own** impulse, not bank `W`

Per-item top-1 under self_generated equals the greedy-generated first answer token (` 1`, ` 8`,
` 4`, ` 0`, …) on **8/8** items. Bank `W` (` 25`, ` 14`, ` 22`, …) is never the model's greedy
continuation from `{a}+{b}=`.

### Typing vs generating the **same** impulse string ≈ no difference

mean score: self_generated **+0.57** vs forced_impulse_match **+0.55** (Δ ≈ **−0.02**).

**Read:** the forced/self gap is **not** about autoregressive vs teacher-forced processing of
the same substring. It is **which token** sits in the impulse slot — bank `W` vs model's greedy
answer.

### Practical freeze for E8

1. **Reject** framing persistence as generic “commitment” independent of input on this domain:
   bank-W carry at \(t^*\) is **W-in-prompt phenomenon**.
2. **Accept** that self-generated wrong-ish answers **do** persist — but as **their own**
   tokens, not the bank contrast pair.
3. **E0–E7 mechanistic results** (write-wave, causal patching at `W`, routing) characterize the
   **forced-error** regime; generalization to self-generated arithmetic is **not** licensed
   without a new bank or prompt family where the model actually emits bank `W`.
4. **E8 is done** for the plan's triviality check on forced `25`.

---

## 3. What E8 does *not* establish

- Whether a **different model / prompt** where greedy gen hits bank `W` would show self-gen
  persistence comparable to forced (not tested — 0/8 items here).
- Causal mechanism of W-in-prompt copying (E4 still applies in forced regime only).
- That self-generated impulses lack internal structure — they persist on their own token; they
  just aren't bank `W`.

---

## 4. Next steps

- **Write-up framing:** distinguish **forced-error persistence** (measured E2–E7) from
  **general commitment** (not supported E8).
- **Phase 4 optional:** E9 formation threshold still valid **within forced regime**; label
  accordingly.
- **Design fork:** new items where greedy gen = bank `W`, or compare self-gen top-1 vs bank W/C
  as matched contrasts per item.

---

## 5. Short takeaway

E8 compared typed bank `W` vs greedy self-generated answers before the revision cue. Forced `W`
yields mean score **+2.71** and 100% top-1 = bank `W`. Self-generated paths yield **+0.57** and
0% bank `W` — the model sticks to **its own** greedy impulse (` 1`, ` 8`, …) instead. Typing the
same self-generated substring gives the same score (Δ ≈ 0). **Bank-W persistence is input-copying
of the forced wrong answer**, not a model self-commitment phenomenon on this prompt family.

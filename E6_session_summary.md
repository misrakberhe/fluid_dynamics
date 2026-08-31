# E6 Session Summary — Operand corruption (residue vs recompute)

**Date:** 2026-08-31  
**Artifacts:** `E6_operand_corruption.py`, `E6_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E6)

Sharpest behavioral test: is \(W\) at \(t^*\) **carried** from the first commitment, or **recomputed**
from the visible second-instance operands?

---

## 1. Working setup

```text
baseline:  {a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
corrupt:   {a} + {b} = {W}. Wait, let me recompute. {a'} + {b'} =
filler:    {a} + {b} = {W}. Wait, let me recompute. 99 + 99 =
```

First-instance wrong answer `W` **unchanged**. Each item uses a **different** valid bank pair
for \((a', b')\) with its own \(W', C'\).

**Primary readout at \(t^*\):** original \(\mathrm{logit}(W) - \mathrm{logit}(C)\) (does the model
still prefer the **first** wrong answer?).

**Secondary:** score on visible pair \(\mathrm{logit}(W') - \mathrm{logit}(C')\); top-1 token.

8 items, GPT-2 small.

---

## 2. Empirical conclusions

### Neither pure residue nor pure recompute

| Condition | mean score_orig (W−C) | frac top-1 = orig `W` |
|---|---|---|
| baseline | **+2.71** | **1.00** |
| corrupt ops2 | **+0.42** | **0.625** |
| filler ops2 | **+0.82** | **0.375** |

**Corrupt Δ vs baseline:** mean **−2.29** (large collapse in original W preference).

**Read:** corrupting visible operands **strongly weakens** sticking on the **original** wrong
answer — so behavior is **not** pure carry from the first `W` token alone.

### But the model does **not** cleanly recompute from visible operands

- mean `score_visible` (W′−C′ for corrupted pair) under corrupt: **+0.06** (near chance).
- frac top-1 = visible \(W'\): **0/8**.

Top-1 under corrupt often flips to **unrelated** tokens (` 16`, ` 6`, ` 99`) — not the answer
implied by \((a', b')\).

**Read:** not a clean switch to recomputing the visible sum; more like **disruption** or
**interference** than correct alternate arithmetic.

### Corrupt > filler specifically (operands matter)

Filler ops2 (`99 + 99`) also hurts (mean score **+0.82**), but **less** than valid corrupt ops2
(**+0.42**). Visible operand **content** pulls preference more than length-matched nonsense.

### Item-level pattern

5/8 items still top-1 = orig `W` under corrupt (soft score often much lower). 3/8 flip top-1
away from orig `W` without landing on visible \(W'\).

### Practical freeze for E6

1. **Reject** “pure residue” (orig `W` unaffected by ops2) and **reject** “pure recompute”
   (model answers visible pair).
2. **Accept** **partial / fragile carry**: first commitment still influences \(t^*\), but visible
   operands materially **modulate** it — consistent with E4 (content at `W` matters) and E5
   (routing not sole channel; visible tokens also in context).
3. **E6 is done** for the sharp residue/recompute dichotomy on this domain.

---

## 3. What E6 does *not* establish

- Mechanism of the collapse (attention to ops2 vs embedding interference).
- Whether softer score alone (“still positive but weaker”) counts as residue — operational
  choice; we report both score and top-1.
- Generality beyond this prompt family.

---

## 4. Next steps

- **E7** cue overwrite toward `C` (does revision span write competing signal?).
- **E8** self-generated vs forced impulse.
- Optional: corrupt **only one** operand, or corrupt to same sum different tokens.

---

## 5. Short takeaway

E6 corrupted second-instance operands while leaving the first `25` intact. Original \(W\)
preference at \(t^*\) **collapses partially** (Δ ≈ **−2.3**; top-1 stays orig `W` on only
**5/8** items). The model does **not** recompute the visible corrupt sum (top-1 never = \(W'\)).
Sticking is **neither** pure carried residue **nor** pure visible recomputation — a **mixed,
fragile carry** modulated by what's in the recompute slot.

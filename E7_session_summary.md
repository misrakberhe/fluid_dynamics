# E7 Session Summary — Overwrite test (does the cue write toward C?)

**Date:** 2026-08-31  
**Artifacts:** `E7_overwrite_test.py`, `E7_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E7)

If the revision cue recruited a **competing overwrite** toward the correct answer, we should see
post-embed writes along \(w = W_U[:,W] - W_U[:,C]\) at the **revision span** that are **negative**
(toward \(C\)). E2 DLA machinery, applied to revision tokens instead of the impulse position.

---

## 1. Working setup

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
```

| Label | Meaning |
|---|---|
| `w` | \(W_U[:,W] - W_U[:,C]\); positive projection → toward `W`, negative → toward `C` |
| revision span | tokens from ` Wait` through `.` before second operands (7 tokens) |
| middle span (control) | length-matched ` ...` filler replacing cue |
| metric | \(\sum_L \sum_{p \in \mathrm{revision}} (\mathrm{attn\_out} + \mathrm{mlp\_out}) \cdot w\) |

8 items, GPT-2 small, `fold_ln` default. Secondary: unit-normalized mean direction at revision;
region totals at `W`, `W_window`, `ops2` for scale.

---

## 2. Empirical conclusions

### No systematic competing push toward `C`

| Metric | Value |
|---|---|
| mean Σ post-embed · w @ revision (cue) | **+17.7** (toward `W`) |
| frac items with net write toward `C` | **3/8** (37.5%) |
| mean unit-normalized direction @ revision | **≈ 0** (−0.0008) |
| mean final score at \(t^*\) | **+2.71** (unchanged; all top-1 = `W`) |

**Read:** on aggregate, the revision cue does **not** write a competing wave toward `C`. Raw
signed totals lean **toward `W`**, and magnitude-normalized direction is **neutral**.

### Item-level sign is noisy; not a clean overwrite signal

| item | Σ revision · w | toward `C`? | score @ \(t^*\) |
|---|---|---|---|
| ops_12+15 | **+71.7** | no | +4.19 |
| ops_8+7 | **−57.3** | yes | +2.44 |
| ops_11+12 | −0.06 | yes | +1.87 |
| ops_9+6 | **−50.8** | yes | +2.77 |
| ops_13+14 | +26.5 | no | +1.26 |
| ops_4+5 | +49.7 | no | +2.11 |
| ops_16+10 | +42.2 | no | +4.49 |
| ops_3+8 | +59.9 | no | +2.52 |

Three items show large negative revision writes (toward `C` along final `w`), yet **all** still
prefer `W` at \(t^*\). Signed revision DLA alone does not predict behavioral flip — same pattern
as E2 at the impulse position.

### Cue vs length-matched filler

- mean revision write (cue): **+17.7**
- mean revision write (filler): **+9.3**
- mean cue − filler: **+8.4** (high variance across items)

The cue **changes** middle-span writes vs filler, but not in a consistent **toward-`C`** direction.

### Regional scale (cue prompt, means)

| region | mean Σ (attn+mlp)·w |
|---|---|
| revision | +17.7 |
| `W` | +2.9 |
| `W_window` | +8.9 |
| ops2 | +37.7 |

Revision writes are nontrivial but **not** dominantly anti-`W`. Operand reprise (`ops2`) carries
the largest post-embed mass along `w`.

### Practical freeze for E7

1. **Reject** “momentum resists the cue” as the primary story — there is **no reliable competing
   overwrite toward `C`** at the revision span.
2. **Accept** “cue is not recruited for correction” — revision tokens do not systematically
   implement an anti-`W` write-wave.
3. Item-level negative revision writes exist but **do not** co-occur with behavioral revision (E1).
4. **E7 is done** for the overwrite-vs-resistance dichotomy on this domain.

---

## 3. What E7 does *not* establish

- Whether ablating revision writes would change \(t^*\) (E4-style causal test on revision; E4
  ablation there was weak).
- Semantic parsing of cue tokens (“recompute” vs “Wait”) — only aggregate DLA.
- Whether a **different** contrast direction or tuned-lens readout at revision would show overwrite.

---

## 4. Next steps

- **E8** — self-generated vs forced impulse (trivial input-copying check).
- Optional: causal patch revision residual toward C-prompt state (stronger overwrite test).
- Update plan pointer: E7 complete — no competing C-wave at revision.

---

## 5. Short takeaway

E7 applied E2 DLA to the revision span. The cue does **not** produce a systematic write toward
`C`: mean signed projection is **+17.7** (toward `W`), unit-normalized direction ≈ **0**, and
only **3/8** items net-negative. Behavior still sticks on `W` everywhere. The right description
is **cue not recruited for overwrite**, not momentum **resisting** a competing correction signal.

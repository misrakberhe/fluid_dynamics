# E1 Session Summary — Concepts, Conclusions, Next Steps

**Date:** 2026-08-12  
**Artifacts:** `E1_behavioral_revision_curve.ipynb` (Design A + soft metrics + Design B)  
**Plan reference:** `momentum-experiment-plan.md` Phase 1 (E1)

This note freezes what was clarified and measured in the E1 behavioral baseline. It is **not** a momentum / impulse / residue result, and it uses **no** mid-layer interpretability.

---

## 1. Working setup

Same forced-error family as E0. Score is always **final next-token behavior** at \(t^*\) (the last ` =`), after the full prompt including cue and recompute operands.

Canonical skeleton:

```text
{a} + {b} = {W}.{MIDDLE} {a} + {b} =
```

| Label | Meaning |
|---|---|
| `W` | wrong answer token (e.g. ` 25`) |
| `C` | correct answer token (e.g. ` 27`) |
| score | \(\mathrm{logit}(W) - \mathrm{logit}(C)\) at \(t^*\) (positive ⇒ prefers W) |
| `p_C_pair` | softmax over {W, C} only: \(e^{\ell_C}/(e^{\ell_W}+e^{\ell_C})\) |
| top-1 | full-vocab argmax at \(t^*\) (what greedy decode would emit) |
| \(t^*\) | final ` =` — **not** the first cue token; the whole cue is already in context |

Model: GPT-2 small via TransformerLens. Final-logit scoring only (`fold_ln` irrelevant here).

Prompt bank: **8** operand pairs with single-token W/C. Cues: `Wait, let me recompute.`, `Actually…`, `Hold on…`.

**What we are not measuring:** preference “during” the cue span. The cue is prompt text; E1 asks what answer the model chooses after everything.

---

## 2. Concepts clarified

### What E1 is for
E1 is the **behavioral baseline** every later structural claim must beat. If “later cue → harder revision” already shows up as a clean curve from token placement alone, interpretability has little left to explain on that axis. If revision never succeeds, the baseline is: *W sticks; cues do not flip the answer*.

### Design A — cue lag
Depth = number of filler tokens (` ...`) between `W.` and the cue. Lag 0 = E0-style adjacency.

Control: length-matched **no_cue** (filler replaces cue tokens).

### Design B — wrong-trajectory depth
Depth = number of **wrong-commitment steps** that restate/reinforce W before the cue (e.g. ` Yes, 25.`, ` So the sum is 25.`, …).

Controls at matched length:

| Condition | Middle |
|---|---|
| `wrong+cue` | wrong steps + cue |
| `wrong+no_cue` | wrong steps + filler matched to cue length |
| `filler+cue` | ` ...` matched to wrong-steps length + cue |

Separates wrong *content* from mere token count.

### Hard vs soft metrics
- **Hard:** top-1 is C (revision success) / is W.
- **Soft:** score, `p_C_pair`, `leans_C` (score < 0). Soft can move when top-1 is saturated.

### Why mid-cue readout is out of scope for E1
We do not want a different *sequence* through the cue; the cue is fixed context. The behavioral question is only the answer token after `=`. Cue-local writes belong later (e.g. E7).

---

## 3. What was tried

### Design A (§1–8)
1. 8 items × 3 cues × lags 0–8 + no_cue control (~288 forwards).
2. Aggregate curves: mean score and frac top-1=C vs lag.
3. Soft dive: `p_C_pair`, cue−no_cue deltas, lag-0 histograms, per-item slopes.

### Design B (§9–10)
1. Same 8 items × depths 0–8 × three conditions (`wrong+cue`, `wrong+no_cue`, `filler+cue`).
2. Soft/hard aggregates + deltas vs filler+cue and vs wrong+no_cue.
3. Per-item slopes under `wrong+cue`.

---

## 4. Empirical conclusions

### Design A — binding hard result
- Under **all cued** conditions and lags: **frac top-1=C = 0**. Greedy answer stays **W**.
- Soft: cues keep mean score strongly positive (~2.1–2.8). Lag softens only slightly (`wait_recompute` slope ≈ **−0.05 score / token**).
- **Cues hurt vs length-matched no_cue:** mean `delta_score` (cue − no_cue) ≈ **+1.3 to +1.8** for all three cues; **0%** of cued rows have `delta_score < 0`.
- Pairwise `p_C_pair` under cues ~0.08–0.14 vs no_cue ~0.26–0.34.
- `leans_C` (score < 0): **0/216** cued rows; **7/72** no_cue (only at higher lag).

**Read:** cue lag is not a useful “revision success vs depth” curve here — there is almost no success. Soft metrics show mild softening with lag and a clear **cue-makes-W-worse** effect vs filler.

### Design B — more interesting soft structure
- Hard: still **frac top-1=C = 0** and **frac leans_C = 0** for every condition/depth.
- `wrong+cue` score slope vs depth ≈ **−0.04 / step** (slightly softer, not harder).
- `wrong+no_cue` score slope ≈ **+0.10 / step** — wrong prose **without** a cue strengthens W preference.
- Content vs length: `wrong+cue` is **below** `filler+cue` on average (mean Δ ≈ **−0.41**; only ~7% of rows worse than filler). Wrong commitment + cue is *not* “more stuck than length-matched filler + cue.”
- Cue after wrong steps still hurts vs `wrong+no_cue` (mean Δ ≈ **+0.40**, ~86% of rows), but the gap shrinks with depth.

**Read:** commitment builds when wrong content accumulates without a cue. Adding the cue does not produce revision; relative to filler+cue it does not amplify W further on soft score. Still no behavioral flips.

### Practical baseline policy (frozen)
1. On GPT-2 small forced-error arithmetic, **expect greedy answer = W** after revision cues.
2. Do **not** frame later experiments as explaining a lag/depth revision-success curve — that curve is flat at zero success.
3. Structural measures (E2+) should beat or explain: *W persists; cues fail / often reinforce W vs filler controls.*
4. Soft score remains useful for graded internal comparisons; hard top-1 is the behavioral claim.

**E1 is done** for the purpose of establishing this baseline. Optional polish (much larger lag spot-check, more cues) is low priority unless a later experiment needs it.

---

## 5. What E1 does *not* establish

- Nothing about where W is written, how it routes to \(t^*\), or whether residue vs recomputation dominates (E2–E6).
- Whether the cue writes toward C internally (E7) — behavior can stay on W either way.
- Generality beyond GPT-2 small and this arithmetic contrast family.
- That “momentum” exists — only that, behaviorally, revision language does not override the forced wrong answer here.

---

## 6. Next steps

### Immediate
1. Treat E1 as closed for Design A + B on this domain.
2. Optionally pointer in `momentum-experiment-plan.md` / `NOTES.md`: E1 complete — no revision success under cues; baseline = W sticks.

### Next experiment: **E2 — Write localization**
- Layer × position attribution of writes onto the W−C direction (`attn_out` / `mlp_out`).
- Question: sharp impulse vs gradual accumulation around the first W.
- Meter policy from E0: prefer **tuned lens** (or final) scoring for mid-depth reads; do not over-interpret early logit-lens heatmaps.
- Interpret against E1: we already know behavior ends on W — ask *where that preference is written*.

### After the spine (plan order)
- E3 persistence / routing at \(t^*\)
- E4 content patching, E5 path ablation
- E6 operand corruption (residue vs recomputation)
- E7 overwrite at the cue (does the cue write toward C at all?)

### Open forks (unchanged)
- J-lens / causal faithfulness still deferred.
- Theory vocabulary stays out of experiment definitions until the empirical spine exists.

---

## 7. Short takeaway

E1 asked whether revision success varies with cue placement (lag) or wrong-trajectory depth. On GPT-2 small forced-error arithmetic, **cues never flip the greedy answer to C**. Soft metrics show cues often **increase** W preference vs length-matched filler; wrong prose without a cue strengthens W with depth. The behavioral baseline is set: **W sticks**. Move to **E2**.

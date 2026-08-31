# E4 Session Summary — Concepts, Conclusions, Next Steps

**Date:** 2026-08-31  
**Artifacts:** `E4_content_patching.ipynb`, `E4_content_patching.py`, `E4_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E4)

This note freezes **causal** evidence about whether the impulse-locus residual state determines
\(W\) preference at \(t^*\). It is **not** routing/path ablation (E5) or operand corruption (E6).

---

## 1. Working setup

Same forced-error family. Two prompt variants per item:

| Variant | Skeleton |
|---|---|
| **W-prompt** | `{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =` |
| **C-prompt** | `{a} + {b} = {C}. Wait, let me recompute. {a} + {b} =` |

| Label | Meaning |
|---|---|
| impulse / `W_window` | forced-answer slot + ±2 tokens (E3 nomination) |
| \(t^*\) | final ` =` |
| score | \(\mathrm{logit}(W) - \mathrm{logit}(C)\) at \(t^*\) |

Model: GPT-2 small, `fold_ln=False`. Prompt bank: **8** items.

**Primary interventions** (layers **L5–L11** unless noted):

| Name | What it does |
|---|---|
| `necessity_resid_Wwin_L5-11_Cswap` | On W-prompt: replace `W_window` `resid_post` with C-run values |
| `sufficiency_resid_Wwin_L5-11_Wpatch` | On C-prompt: patch W-run `resid_post` into `W_window` |
| `ablate_writes_Wwin_L5-11` | On W-prompt: zero `attn_out` + `mlp_out` at `W_window` |
| Per-layer sweep | Single-layer `resid_post` C-swap at `W_window` |

**Controls:** ablate writes at `revision`, `ops2`, distance-matched random position.

---

## 2. Concepts clarified

### What E4 is for
E3 nominated **where** \(t^*\) reads from (W neighborhood, L5–L11). E4 asks whether that
state **causally matters** — not just correlates.

### Residual swap vs write ablation
- **Residual swap** replaces the full accumulated state at a site with a counterfactual run's state.
- **Write ablation** zeros only that layer's new `attn_out`/`mlp_out` contribution at a site.
  Earlier layers' accumulated content remains. These can diverge sharply.

### Necessity vs sufficiency
- **Necessity:** without W-locus state (C-swapped), does W preference at \(t^*\) collapse?
- **Sufficiency:** on a C-prompt, does injecting W-locus state restore W preference?

### Counterfactual
C-prompt uses the **correct** first-answer token at the impulse slot. Same structure, different
committed answer — isolates the forced-error state rather than a unrelated clean prompt.

---

## 3. What was tried

1. Band residual C-swap at `W_window` (necessity) and W-patch (sufficiency), L5–L11 and L9–L11.
2. W-only residual C-swap (vs full window).
3. Write ablation at `W`, `W_window` (L5–L11, L9–L11).
4. Per-layer necessity sweep (L0–L11, one layer at a time).
5. Region controls: `revision`, `ops2`, random position write ablation.

---

## 4. Empirical conclusions

### Residual at W_window is **causally necessary and sufficient**
- **Necessity** (`necessity_resid_Wwin_L5-11_Cswap`): mean Δ score = **−4.94** (baseline +2.71 →
  patched −2.23). Wipes most of W preference.
- **Sufficiency** (`sufficiency_resid_Wwin_L5-11_Wpatch`): mean Δ = **+4.92** (C-baseline −2.68 →
  patched +2.24). Nearly fully restores W preference on the C-prompt.
- **W-only** C-swap (L5–L11): Δ = **−4.84** — impulse token alone carries almost all the effect.

**Read:** the residual state at the forced-error locus **causally determines** \(t^*\)'s W
preference. This is the strongest premise-supporting result in the spine so far.

### Write ablation is **much weaker** than residual swap
- `ablate_writes_Wwin_L5-11`: mean Δ = **−0.30**
- `ablate_writes_W_L5-11`: Δ = **−0.22**
- `ablate_writes_Wwin_L9-11`: Δ = **−0.16**

**Read:** zeroing late writes does not remove the accumulated state that \(t^*\) reads. The
**content** at the locus matters; individual layer writes are not the right surgical unit for
ablation (redundancy / already-integrated state). E4 patching targets should be **residual
states**, not single write vectors.

### Per-layer necessity: **mid-depth L5–L8** dominates; L9–L11 minimal
| Layer | mean Δ (single-layer C-swap) |
|---|---|
| L0–L4 | ~−5.4 |
| L5 | −4.94 |
| L6 | −4.32 |
| L7 | −2.67 |
| L8 | −2.31 |
| L9 | −0.13 |
| L10–L11 | ~0 |

**Read:** causal leverage concentrates where the write-wave **integrates** (L5–L8), not where
E3 saw peak OV **routing** (L9–L11). Writing and reading are depth-separated: late layers
route from state already fixed mid-depth.

### Controls
- `ablate_writes_revision_L5-11`: Δ = **−0.02** — negligible.
- `ablate_writes_ops2_L5-11`: Δ = **−0.87** — nontrivial, but `randpos` control Δ = **−0.84** —
  likely nonspecific damage, not specific ops2 routing.
- L9–L11 band C-swap alone: Δ = **−0.13** — confirms late resid swap barely matters.

### Practical freeze for E4
1. **Causal claim is supported:** W-locus residual state determines \(t^*\) preference.
2. **Patch unit for E4+/E6:** `resid_post` at `W_window`, layers **L5–L8** (primary), not
   single write vectors.
3. **Do not** interpret weak write ablation as absence of causality — it's a method mismatch.
4. **Do not** over-interpret ops2 write ablation given randpos control.

**E4 is done** for the core causal question on this domain.

---

## 5. What E4 does *not* establish

- Whether the **route** from W to \(t^*\) is necessary (E5 path ablation) — content and routing
  can come apart.
- Whether preference is **carried** vs **recomputed** from visible operands (E6).
- Redundant backup paths fully characterized — a single-site write ablation null would not have
  been informative anyway.
- Generality beyond GPT-2 small and this arithmetic family.

---

## 6. Next steps

### After E4b (probe-local patching)
E4b patched **`t*`'s own residual** (not `W_window`). Early layers at `t*` (L0–L4) do **not**
decide the answer (Δ ≈ 0); late layers at `t*` (L9–L11) do (Δ ≈ **−5.4**). This inverts E4's
depth profile at `W` (mid integrates, late routes). See `E4b_session_summary.md`.

### Immediate: **E5 — Path ablation**
- Block attention \(t^* \to\) `W_window` vs \(t^* \to\) `ops2`, leaving impulse writes intact.
- Test whether routing or content (E4) is the binding constraint.
- Priority heads from E3: L9H9, L11H0, L10H7.

### Then
- **E6** operand corruption — now well-motivated: E4 says state matters; E6 asks if it's
  *carried* when operands change.
- **E9** scaled patching (depends on E4 machinery) — sweep α on W-locus resid injection.

### Open forks (unchanged)
- J-lens / causal faithfulness deferred.
- Theory vocabulary stays out of metric definitions.

---

## 7. Short takeaway

E4 asked whether the impulse-locus state causally determines \(W\) at \(t^*\). **Yes.** Swapping
`W_window` residuals to the C-counterfactual collapses W preference (Δ ≈ **−5**); patching
W-run residuals into the C-prompt restores it (Δ ≈ **+5**). Single-layer write ablation is weak
(Δ ≈ **−0.3**) because accumulated state persists. Causal leverage sits at **L5–L8 integration**,
not L9–L11 routing. Move to **E5** to test whether the **path** is also necessary.

# E5 Session Summary — Path ablation (routing at t*)

**Date:** 2026-08-31  
**Artifacts:** `E5_path_ablation.py`, `E5_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E5)

E5 asks whether **attention routes** from \(t^*\) to nominated sources matter for the final W
preference — with impulse **writes left intact** (unlike E4 content patching).

---

## 1. Working setup

W-prompt, 8-item bank. **Intervention:** at selected layers, zero attention from query \(t^*\) to
source key positions, **renormalize** remaining keys per head.

| Intervention | Layers | Block \(t^* \to\) |
|---|---|---|
| `path_block_Wwin_L5-11` | L5–L11 | `W_window` |
| `path_block_Wwin_L9-11` | L9–L11 | `W_window` |
| `path_block_W_only_L9-11` | L9–L11 | `W` only |
| `path_block_ops2_L9-11` | L9–L11 | reprised operands |
| `path_block_revision_L9-11` | L9–L11 | cue span |
| `path_block_randpos_L9-11` | L9–L11 | distance-matched control |
| `path_block_Wwin_L{L}H{h}` | single head | `W_window` (E3 priority heads) |
| Per-layer sweep | L0–L11 | `W_window` |

Readout: \(\mathrm{logit}(W)-\mathrm{logit}(C)\) at \(t^*\).

---

## 2. Concepts clarified

### Content (E4) vs routing (E5)
- **E4** changes the **state** at `W` (residual swap) — "does that representation matter?"
- **E5** blocks **reading** that state from \(t^*\) — "does that route matter?"
- They can dissociate: content can matter while a single route is redundant, or vice versa.

### Renormalization
After zeroing \(t^* \to\) source keys, remaining attention weights are renormalized so each
head's distribution still sums to 1. The model can reroute through other positions.

---

## 3. Empirical conclusions

### Routing to `W_window` matters — but less than content replacement (E4)
| Intervention | mean Δ score |
|---|---|
| E4 ref: `necessity_resid_Wwin_L5-11_Cswap` | **−4.94** |
| `path_block_Wwin_L5-11` | **−2.30** |
| `path_block_Wwin_L9-11` | **−0.62** |
| `path_block_W_only_L9-11` | **−0.65** |

**Read:** blocking \(t^* \to\) `W_window` **hurts** W preference (routing is **necessary**), but
the effect is **~half** the E4 content swap over L5–L11 and **~8× smaller** for the late band
alone. The committed **state at `W`** (E4) binds more than any single **route** (E5) — consistent
with distributed landing and rerouting backup paths.

### Late nominated band (L9–L11) still contributes
- `path_block_Wwin_L9-11`: Δ = **−0.62** — nontrivial despite E4b showing late `t*` resid is
  causally load-bearing (may read `W` early in the layer stack then integrate locally).
- Top single head: **L9H9** Δ = **−0.53**; L11H0 Δ ≈ **−0.01** (weak).

### Operand and revision routes are not specific at late layers
- `path_block_ops2_L9-11`: Δ = **−0.07**
- `path_block_revision_L9-11`: Δ = **−0.03**
- `path_block_randpos_L9-11`: Δ = **−0.06**

**Read:** late \(t^* \to\) `ops2` / revision blocks are indistinguishable from random-position
control — not specific routes for sticking on `W` at L9–L11 (cf. E3 OV mass to `W_window`).

### Per-layer path block \(t^* \to\) `W_window` (mean Δ)
Peaks **late** (L10–L11 largest in sweep) — aligns with E3/E4b late landing, though band
ablation L5–L11 captures most total effect.

### Practical freeze for E5
1. **Routing to `W_window` is necessary** but **not sufficient** to explain sticking — content
   at `W` (E4) is the stronger binding constraint.
2. **Late path block** (L9–L11) has modest specific effect; **mid+late band** (L5–L11) is stronger.
3. Do **not** treat ops2/revision path blocks as evidence for operand-based recomputation at late
   depth — null vs randpos control.
4. **E5 is done** for the routing question on this domain.

---

## 4. What E5 does *not* establish

- Whether **all** routes are characterized (only nominated bundles tested).
- Operand corruption / residue vs recompute (E6) — sharper behavioral test.
- Generality beyond GPT-2 small forced-error arithmetic.

---

## 5. Next steps

- **E6** operand corruption — direct residue vs recompute test (now well motivated: E4 content
  > E5 routing).
- **E7** cue overwrite toward `C`.
- Optional: combine E4+E5 — path block after partial content patch.

---

## 6. Short takeaway

E5 blocked attention from \(t^*\) to nominated sources while leaving writes intact. Blocking
\(t^* \to\) `W_window` **reduces** W preference (Δ up to **−2.3** over L5–L11; **−0.62** late
only) but **less than** replacing the `W` state in E4 (Δ ≈ **−4.9**). Late blocks to `ops2` /
revision match random controls. **Content at the commitment locus matters more than any single
route** — residue is not *only* routing, but routing to `W_window` is still part of the story.

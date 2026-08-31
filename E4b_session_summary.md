# E4b Session Summary — Probe-local patching at \(t^*\)

**Date:** 2026-08-31  
**Artifacts:** `E4b_content_patching_tstar.ipynb`, `E4b_content_patching_tstar.py`, `E4b_outputs/`  
**Extends:** E4 (source patching at `W_window`)

E4 showed the **source state at `W`** causally determines \(t^*\)'s answer. E4b asks the
complementary question: is the answer already **baked into early layers at `t*` itself**, or
decided only after late processing **at the probe position**?

---

## 1. Working setup

Same W-prompt / C-prompt pair as E4. Interventions patch **`resid_post` at position `t*`** (the
final `=`), not at `W_window`.

| Intervention | Meaning |
|---|---|
| `necessity_resid_tstar_{band}_Cswap` | On W-prompt: replace `t*`'s residual with C-run's `t*` residual |
| `sufficiency_resid_tstar_L5-11_Wpatch` | On C-prompt: inject W-run's `t*` residual |
| `ablate_writes_tstar_L5-11` | Zero `attn_out`+`mlp_out` at `t*` |
| `ref_E4_necessity_Wwin_L5-11_Cswap` | Same-item E4 reference (patch `W_window`) |

Bands: L0–4 (early), L5–8 (mid), L9–11 (late), L5–11, L0–11. Plus per-layer sweep L0–L11.

---

## 2. Key conceptual distinction

| Site patched | Question answered |
|---|---|
| **E4: `W_window`** | Is the committed **source representation** necessary? |
| **E4b: `t*`** | Is the answer already in **`t*`'s own early residual**, or decided late at `t*`? |

These use **different positions** in the same forward pass. Layer L at `W` ≠ layer L at `t*`.

---

## 3. Empirical conclusions

### Early layers at `t*` do **not** decide the answer
- `necessity_resid_tstar_L0-4_Cswap`: mean Δ = **−0.01** (negligible).
- Per-layer L0–L4 at `t*`: each |Δ| < **0.01**.

**Read:** despite causal attention letting `t*` see all prior tokens from layer 0, **`t*`'s own
early residual does not yet encode the final W vs C commitment.** Prior tokens are visible but not
yet locally integrated into a decisive probe state.

### Late layers at `t*` are **causally decisive**
- `necessity_resid_tstar_L9-11_Cswap`: mean Δ = **−5.39**
- `necessity_resid_tstar_L5-11_Cswap`: Δ = **−5.39** (same — effect saturates by L9)
- Per-layer ramp: L5 −0.47 → L6 −1.06 → L7 −2.73 → L8 −3.05 → **L9 −5.28 → L11 −5.39**

**Sufficiency** (`sufficiency_resid_tstar_L5-11_Wpatch`): Δ = **+5.39** — symmetric restoration.

**Read:** the final preference is **fixed late at the probe position** (L9–L11), not in `t*`'s
first few layers.

### Inverted depth profile vs E4 (`W_window` patching)

| Layer band | E4: patch `W_window` | E4b: patch `t*` |
|---|---|---|
| L0–L4 | ~−5.4 (large) | ~0 |
| L5–L8 | −4.9 → −2.3 | −3.1 (band) |
| L9–L11 | ~0 | **−5.4** |

**Read:** **writing and reading are depth-separated at different positions:**
- `W` **integrates** commitment mid-depth (L5–L8; E4).
- `t*` **decides** late (L9–L11; E4b), plausibly by reading that committed source.

This directly answers the concern that "previous tokens are baked into `t*`'s first layers" — they
are **accessible** early but **not sufficient**; late `t*` processing is still causally necessary.

### Write ablation at `t*` is stronger than at `W`
- `ablate_writes_tstar_L5-11`: Δ = **−2.58**
- E4 `ablate_writes_Wwin_L5-11`: Δ = **−0.30**

**Read:** ablating what `t*` is **actively computing** in late layers matters more than ablating
late writes at the distant source — consistent with decision happening **at the probe**, not
only at `W`.

### E4 source result unchanged
- `ref_E4_necessity_Wwin_L5-11_Cswap`: Δ = **−4.94** (replicated on same forward pass).

Both are necessary: **committed state at `W`** (E4) **and** **late processing at `t*`** (E4b).

---

## 4. Practical freeze

1. **Do not** infer from E4 alone that `t*` decides early — E4b refutes that.
2. **Mechanistic picture:** mid-depth integration at `W` → late readout/computation at `t*`.
3. **E5 path ablation** should target **late layers at `t*` → `W_window`** (L9–L11 heads from E3),
   not early `t*` layers.
4. E4b is **done** for the probe-local timing question.

---

## 5. Follow-up: readout vs causality (E4b extension)

**Artifact:** `E4b_readout_vs_causality.py`, plots in `E4b_outputs/readout_vs_causality_*.png`

Plots **two curves at `t*`** on the same depth axis:

| Curve | What it measures |
|---|---|
| **Tuned-lens W−C** | When does preference for W *appear* in the meter? (E0 readout) |
| **Patch Δ (C-swap)** | When is `t*`'s residual *causally necessary* for the final score? |

### Readout precedes causal peak — they are not the same thing

| Metric | Mean depth |
|---|---|
| Emergence (tuned-lens, W−C > 0 and stays > 0) | **~L4** (item range L0–L6) |
| Causal onset (\|patch Δ\| > 1 at single layer) | **~L6–L7** |
| Causal peak (max \|patch Δ\|) | **~L10** |

Example mean values at `t*`:

| Layer | Tuned-lens W−C | Patch Δ |
|---|---|---|
| L3 | +0.19 | −0.01 |
| L5 | +0.15 | −0.47 |
| L7 | +1.36 | −2.73 |
| L9 | +2.49 | −5.28 |
| L11 | +3.08 | −5.39 |

**Read:** at L3–L5, the tuned lens can already read weak W preference at `t*`, but swapping `t*`'s
residual barely moves the **final** score (later layers recompute). By L9, readout is strong **and**
the state is causally locked in. This is the operational gap behind "locally decisive" — we should
say **readout emergence ≠ causal fixation**.

### J-lens connection (deferred; directional proxy added)

Full [Anthropic J-lens](https://github.com/anthropics/jacobian-lens) (`unembed(J_l @ h)` with
corpus-averaged Jacobian) was **not** fitted here: no pretrained GPT-2 lens, `jlens` requires
`transformers>=5.5` (conflicts with current env), and fitting is ~96 backward passes/prompt.

**Proxy added:** gradient **∂(W−C)/∂h · ŵ** at `t*` per layer (local directional sensitivity). Plotted on
`readout_vs_causality_raw.png`. This is **not** full J-lens transport (`unembed(J_l @ h)`); it
measures infinitesimal sensitivity along ŵ only, so it ramps modestly (≈0.04→0.10) while full
C-swap patch Δ explodes late (−5.4). Still: tuned-lens emerges **earlier** than either causal
measure — the qualitative J-lens warning holds.

| Layer | Tuned-lens | ∂score/∂h·ŵ | Patch Δ |
|---|---|---|---|
| L3 | +0.19 | +0.04 | −0.01 |
| L5 | +0.15 | +0.07 | −0.47 |
| L9 | +2.49 | +0.10 | −5.28 |

---

| Say | Don't say |
|---|---|
| "W preference is **readable** at `t*` from ~L4–L6" | "`t*` has **decided** by L4" |
| "Final score is **causally tied** to late `t*` state (L9–L11)" | "The answer is **determined** at L9" |
| "Early `t*` state is **causally redundant**" | "Early `t*` has no information" |

---

## 6. Short takeaway

Patching `t*`'s own residual shows the answer is **not** causally fixed in `t*`'s early layers
(L0–L4 patch: Δ ≈ 0), even though the tuned lens can **read** weak W preference there. Causal
leverage ramps mid–late (peak L9–L11: Δ ≈ **−5.4**). **Readout emergence (~L4) precedes causal
fixation (~L10)** — appearing in the meter is not the same as being necessary for the final
answer. Meanwhile E4's mid-depth integration at `W` remains necessary. Move to **E5**.

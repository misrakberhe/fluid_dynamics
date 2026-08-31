# E0 Session Summary — Concepts, Conclusions, Next Steps

**Date:** 2026-08-11  
**Artifacts:** `E0_readout_validity.ipynb` (first attempt), `E0_tuned_lens_validity.ipynb` (authoritative redo + §9 variants)  
**Plan reference:** `momentum-experiment-plan.md` Phase 0 (E0 / E0b)

This note freezes what was clarified and measured in the E0 calibration work. It is **not** a momentum / impulse / residue result.

---

## 1. Working setup (forced-error prompt)

Canonical prompt:

```text
12 + 15 = 25. Wait, let me recompute. 12 + 15 =
```

| Label | Meaning |
|---|---|
| `W` | wrong answer token (e.g. ` 25`) |
| `C` | correct answer token (e.g. ` 27`) |
| score | \(\mathrm{logit}(W) - \mathrm{logit}(C)\) |
| `w` | residual-space direction \(W_U[:,W] - W_U[:,C]\) (used for ridge / random controls) |
| impulse | first wrong-answer token in the prompt |
| \(t^*\) | final ` =` (probe position) |
| revision span | cue text between the wrong answer and the recompute |

Model / lenses for the redo notebook:

- GPT-2 small via TransformerLens with **`fold_ln=False`** (required for the Tuned Lens bridge)
- Package **LogitLens** and pretrained **TunedLens** (`lens_resource_id="gpt2"`)
- Reads **`resid_pre`** at each layer, then unembeds through the chosen lens

---

## 2. Concepts clarified

### What E0 is for
E0 calibrates the **readout instrument**, not the theory. Later heatmaps and emergence claims assume mid-layer residual reads mean something like “current preference for W vs C.” If that meter is bad, depth stories are uninterpretable.

### Logit lens
Treat a mid-layer residual as if it were the final residual: apply LayerNorm / unembed (or project onto `w`) and read logits. Cheap and convenient. Problem: mid-layer geometry need not line up with the final unembedding axes, especially early/mid depth.

### Tuned lens
Per-layer affine translators trained (KL to the model’s final next-token distribution), then unembed. Better mid-depth meter when a pretrained lens exists. In this project it is treated as the **authoritative** comparison target for whether the logit lens is usable.

### Residual vs unembedding direction
Final residual and `w` live in the same vector space. The issue is not “wrong object type”; it is whether mid-layer residuals already encode the answer along axes that the final unembed (or a translator) can read.

### Emergence (operational definition used here)
First depth index at which the W−C score at \(t^*\) becomes positive **and stays positive** through the final depth point. Used only to check whether logit and tuned lenses agree on *when* W-preference appears — not yet a causal claim about impulse.

### Ridge / DIY residual→residual translator
Affine map fitted from mid-layer `resid_pre` toward final residual (or scored via full unembed). Useful as a **diagnostic**, not a substitute for a KL-trained tuned lens unless fit quality and agreement are excellent.

### Thresholds (pre-registered style)
A layer is **OK** for logit lens ≈ tuned lens if:

- Pearson corr of W−C scores across sequence positions ≥ **0.90**
- Sign agreement across positions ≥ **0.90**

**Stable-OK** (cross-prompt): OK on **every** prompt variant.

---

## 3. What was tried

### First notebook (`E0_readout_validity.ipynb`)
Compared logit lens to a DIY ridge residual→residual translator on ~263 tokens. Every layer looked SUSPECT. Diagnosis: **setup failure** — \(N \ll d_{\mathrm{model}}=768\), so the ridge penalty dominated. Not a clean scientific verdict on the logit lens.

### Redo notebook (`E0_tuned_lens_validity.ipynb`)
1. Primary: logit lens vs **pretrained Tuned Lens** on the baseline forced-error prompt  
2. E0b: residual-norm growth (sanity that raw projections can track size)  
3. Secondary: large-N WikiText-2 ridge translators (~25k tokens) with held-out R² — diagnostic only  
4. Control: norm-matched random directions  
5. Verdict cell with OK/SUSPECT vs tuned lens  
6. **§9:** same primary comparison on 5 prompt variants (operands + cue phrasing)

---

## 4. Empirical conclusions

### Baseline prompt (§3 / §7)
- Emergence: both lenses say **L5** (agree).
- Late-layer **correlation** is excellent (~0.95–0.999).
- OK layers on baseline alone: **L0–L1, L6–L8**; SUSPECT elsewhere (L2–L5; L9–L11 fail mainly on sign ≈ 0.89 despite ~1.0 corr).
- A printed “L0 through L8” contiguous range would be **misleading** — the OK set is non-contiguous.
- WikiText ridge improves fit diagnostics with depth but does **not** replace the tuned lens as the meter of record.

### Prompt variants (§9) — the binding result
Five forced-error variants (baseline, `8+7`, “Actually…”, “Hold on…”, `11+12`):

| Variant | Emergence (logit / tuned) | Example OK layers |
|---|---|---|
| baseline | L5 / L5 agree | 0, 1, 6, 7, 8 |
| ops_8+7 | L6 / L6 agree | 2, 3, 4, 5, 9, 10, 11 |
| cue_actually | L6 / L6 agree | 2, 4, 5, 7, 9 |
| cue_hold_on | L1 / L1 agree | 5, 6, 7, 8, 10, 11 |
| ops_11+12 | L6 / L6 agree | 2, 4, 6, 7, 8, 9, 10, 11 |

Cross-prompt:

- **Emergence always agrees** between logit and tuned lens (depths differ by prompt: L5, L6, or L1).
- **No layer is stable-OK** under corr≥0.9 and sign≥0.9 on all five variants.
- Closest: **L7 at 4/5**. Most layers ≤3/5.
- Failures are often **sign agreement**, not correlation — late layers can have corr ≈ 1 with `min_sign` dipping to ~0.81–0.89.
- The single-prompt trustworthy band **does not generalize**.

### Practical meter policy (frozen)
1. **Primary meter for E1 / E2+:** pretrained **tuned lens** W−C scores.  
2. **Do not freeze a logit-lens depth policy** from one prompt.  
3. Logit lens may still be used as a cheap secondary check where a given prompt’s §3 curve agrees; do not hang claims on it alone.  
4. Ridge translator remains diagnostic only.

**E0 is done** for the purpose of de-risking the readout. Further E0 polishing is optional unless a later experiment reveals a lens bug.

---

## 5. What E0 does *not* establish

- Nothing about impulse, residue, momentum, standing waves, or formation thresholds.
- Causal faithfulness of either lens (that is closer to E4 / J-lens-style work).
- Generality beyond GPT-2 small and this W−C arithmetic contrast family.
- That emergence depth *is* the impulse locus — only that both meters agree on when W preference appears at \(t^*\).

---

## 6. Next steps

### Immediate
1. Treat tuned lens as default in new notebooks (`fold_ln=False`, same TunedLens load pattern).  
2. Optionally add a one-line pointer in `momentum-experiment-plan.md` or `NOTES.md` that E0 completed with “primary = tuned lens; no stable logit-lens band.”

### Next experiment: **E1 — Behavioral revision curve**
- No interpretability required.
- Vary where / whether a revision cue appears; score whether the answer flips from W toward C.
- Purpose: baseline every structural claim must beat (token index alone vs structural measures later).

### Then: **E2 — Write localization**
- Layer × position attribution of writes onto the W−C direction (attn_out / mlp_out), preferably scored consistently with the tuned-lens policy where mid-depth reads matter.
- Question: sharp impulse vs gradual accumulation.
- Read results with the E0 caveat: prefer tuned-lens (or final) scoring; do not over-interpret early logit-lens heatmaps.

### After the spine (plan order, not all blocked on E1)
- E3 persistence / routing at \(t^*\)
- E4 content patching, E5 path ablation
- E6 operand corruption (residue vs recomputation)

### Open forks (unchanged)
- J-lens / causal faithfulness still deferred.
- Theory vocabulary (Se/Ni, vortex, standing wave) stays out of experiment definitions until the empirical spine exists.

---

## 7. Short takeaway

E0 asked whether a cheap mid-layer W−C readout can be trusted. Against a real pretrained tuned lens, **agreement is prompt-dependent and no depth is universally OK for the logit lens**. Use the **tuned lens as the primary meter** going forward. Calibration is complete; move to **E1**, then **E2**.

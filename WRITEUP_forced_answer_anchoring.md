# Write-up Draft — Priority A: Forced Answer Anchoring (+ Qwen Dissociation)

**Status:** blog-post track preferred for MATS deadline  
**Canonical prose:** [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md) (~2k words)  
**Evidence base:** E0–E8 (+ E4b) GPT-2 small; Qwen3.5-4B behavior + causal C-swap  
**Date:** 2026-09-03 (Qwen promoted; stats + citations filled)

Related: [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md), [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md)

**Structural note:** Lead with the Qwen behavior–causality dissociation, then walk back to the GPT-2 mechanism. Do **not** expand E9–E11 or self-generated redesign before deadline.

---

## Working title options

1. **Behavioral revision can mask an anchoring mechanism** (Qwen-hook; preferred for blog)
2. **How a typed-in wrong answer persists through a revision cue** (descriptive / GPT-2-first)
3. **Answer-slot anchoring:** causal structure of context-bound arithmetic preferences (mech interp)
4. **Not momentum: mechanistic persistence of a forced answer token in a transformer**

---

## Abstract (draft ~150 words)

We study a minimal setting where a wrong arithmetic answer appears in context and a revision cue
may or may not change the model's next-token preference. On GPT-2 small, revision cues never flip
greedy output to the correct answer (E1). Using residual patching we show mid-depth state at the
answer locus is necessary and sufficient for later preference (E4, Δ ≈ ±5). Scope is typed-in
anchoring, not generic self-commitment (E8). On Qwen3.5-4B, revision **succeeds** behaviorally
(0% top-1 = W) while the same class of mid-depth C-swap still shifts preference (Δ ≈ −2.1,
bootstrap 95% CI [−2.43, −1.69]; rand control ≈ 0)—a **behavior–causality dissociation**. We
frame GPT-2 results as **forced answer anchoring**, and the Qwen result as evidence that
revision can **mask rather than remove** an anchoring lever.

---

## 1. Introduction — what problem is this?

### 1.1 Motivation

Language models often see **provisional answers** in context: user messages, chain-of-thought
drafts, earlier sentences in a document, or in-context examples. A later token position may need
to **continue** or **revise** that answer. When revision language ("wait, let me recompute") is
present but behavior does not change, is that:

- **Resistance** — an internal commitment fighting the cue?
- **Non-recruitment** — the cue never engages a correction pathway?
- **Input anchoring** — the model propagates what already appears in context?

We built an experiment spine to distinguish these, using a stripped-down arithmetic template.

### 1.2 The prompt

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
```

- **W** = a pre-chosen wrong-answer token (e.g. ` 25` for 12+15)
- **C** = paired correct token (e.g. ` 27`)
- **t*** = final `=`; we score logit(W) − logit(C) and greedy top-1

The wrong answer **W is typed into the prompt** at the first answer slot. This is the
**forced-error / forced-answer** setup.

### 1.3 What we are *not* claiming

We do **not** claim generic "momentum," standing waves, or that the model independently commits to
wrong arithmetic and resists revision. E8 bounds scope: bank-W persistence on this bank largely
requires W in the input. The contribution is a **mechanistic account of typed-in answer anchoring**.

---

## 2. Research questions (reframed after E0–E8)

| # | Question | Main evidence |
|---|---|---|
| RQ1 | Does revision language change the answer at t*? | E1: no (0% top-1 = C) |
| RQ2 | Where is preference for W written? | E2: late write-wave at answer locus |
| RQ3 | Where does t* read from? | E3: W_window; mid–late depth |
| RQ4 | Is answer-locus state **causal**? | E4: residual swap Δ ≈ ±5 |
| RQ5 | When is t* causally fixed vs merely readable? | E4b: readout L4, causality L9–11 |
| RQ6 | Does routing or content matter more? | E5: content > routing |
| RQ7 | Carried residue or visible recomputation? | E6: mixed / fragile |
| RQ8 | Does the cue write toward C? | E7: no |
| RQ9 | Is persistence input-copying or commitment? | E8: bank W needs typing |

---

## 3. Methods (summary for write-up)

**Model:** GPT-2 small via TransformerLens.

**Bank:** 8 single-token W/C pairs (see `E4_content_patching.py` ITEMS).

**Primary behavioral metric:** score = logit(W) − logit(C) at t*.

**Secondary behavioral:** top-1 at t*; p_C_pair = softmax mass on C among {W,C} (E1, E6).

**Readout:** tuned lens primary (E0); logit lens secondary.

**Attribution:** DLA — project attn_out and mlp_out onto w = W_U[:,W] − W_U[:,C] (E2, E7).

**Causal:** residual patching and write ablation at nominated loci (E4, E4b); path ablation at t*
(E5).

**Controls:** length-matched no-cue filler; distance-matched positions; C-prompt counterfactual.

Full protocols: `E{N}_session_summary.md` and `E{N}_*.py`.

---

## 4. Results — narrative for the main text

### 4.1 Behavioral floor: revision fails (E1)

Across cue phrasings and insertion depths, greedy decode at t* **always** emits bank W. Soft
metrics move (cue vs filler changes score and p_C_pair), but there is no revision success and no
score < 0 under cued conditions. **Takeaway:** the phenomenon to explain is **persistence**, not
partial revision.

### 4.2 Writes: late distributed wave, not a single kick (E2)

Preference along the W−C direction is added **post-embedding** in late layers (roughly L6–L11),
via both attention and MLP, spatially distributed around the answer locus—not a single mid-depth
spike at one token. Embedding alone is insufficient (|post| ≫ |embed|).

### 4.3 Landing: t* reads from W_window (E3)

Tuned-lens W−C at t* emerges mid-depth and strengthens late. OV-weighted attention from t* to
source regions weights **W_window** highest among nominated bundles (W, revision, ops2).

### 4.4 Causal core: W_window residual determines t* (E4)

Swapping W_window residual from a C-prompt run onto a W-prompt **collapses** W preference
(Δ ≈ −4.9). Patching W_window from W-run onto C-prompt **restores** it (Δ ≈ +4.9). Per-layer
sweep: strongest leverage **L5–L8**, not the latest layers.

Write ablation at W_window is weak (Δ ≈ −0.3)—accumulated state persists without that layer's
new writes. **Patch the state, not individual writes.**

### 4.5 Readable early, fixed late (E4b)

At t*, tuned lens can read weak W preference from ~L4, but patching t*'s own residual only bites
late (L9–L11). **Readout ≠ causal necessity.**

### 4.6 Routing helps; content binds (E5)

Blocking attention t* → W_window reduces W preference (Δ ≈ −2.3 over L5–11) but less than
content swap (E4). Revision and ops2 blocks ≈ controls.

### 4.7 Fragile carry, not pure residue or recompute (E6)

Corrupting second-instance operands while leaving first W intact drops score (Δ ≈ −2.3) and
top-1 stays W on only 5/8 items. Model does not answer the visible corrupt sum. Visible context
**modulates** anchoring without fully resetting or cleanly recomputing.

### 4.8 Cue not recruited for correction (E7)

DLA on revision span: no systematic write toward C; mean projection toward W or neutral.
**Wrong story:** "momentum resists the cue." **Better:** cue does not engage a competing overwrite.

### 4.9 Scope gate: typing matters (E8)

Forced bank W: mean score +2.71, 100% top-1 = bank W. Self-generated greedy answer before cue:
+0.57, 0% bank W, 100% top-1 = gen token. Typing the same string as self-gen ≈ self-gen (Δ ≈ 0).
**Bank-W anchoring is specific to having bank W in the prompt** on this bank.

---

## 5. Mechanistic model (one paragraph + schematic)

**Prose model:** When bank W appears in the answer slot, late layers write a distributed
answer-locus representation along the W−C axis. Mid-depth layers (L5–L8) integrate this into
residual state at W_window that **causally sets** the logit contrast at t*. Late layers at t*
(L9–L11) are load-bearing for the final preference. t* attends to W_window with nontrivial OV
weight; blocking that path weakens but does not abolish sticking. The revision cue does not add
competing C-directed writes; operand changes in the recompute slot modulate but do not replace
the anchored token. Persistence of **this specific W** requires W in the input (E8).

**Schematic (for a figure):**

```text
[operands] = [W typed] . [revision cue] [operands] = [t* probe]
                |                              ^
                v                              |
         late write-wave (E2)                  |
                |                              |
                v                              |
         W_window state (E4: L5-8) ----------+---- path attn (E5, weaker)
         [causal bottleneck]                   |
                |                              |
                v                              |
         late computation at t* (E4b: L9-11) -+
```

---

## 6. Key numbers table (for slides / paper)

| Quantity | Value | Source |
|---|---|---|
| Mean score at t* (forced) | +2.71 | E1/E3/E8 |
| frac top-1 = W (forced, cued) | 100% | E1 |
| Residual C-swap Δ at W_window | −4.94 | E4 |
| Residual W-patch Δ on C-prompt | +4.92 | E4 |
| Write ablation Δ at W_window | −0.30 | E4 |
| Path block t*→W_window Δ (L5–11) | −2.30 | E5 |
| Operand corrupt Δ score | −2.29 | E6 |
| Forced vs self-gen Δ score | −2.13 | E8 |
| Revision span mean write · w (cue) | +17.7 (toward W) | E7 |

---

## 7. Claims checklist

### Safe to claim

- [x] Typed-in wrong answer at the first answer slot causally influences preference at a later `=`.
- [x] Effect is localized to answer-locus residual (W_window), strongest mid-depth (L5–L8).
- [x] Late layers at t* are causally load-bearing despite earlier readable preference.
- [x] Routing to W_window contributes; content swap is the stronger intervention.
- [x] Revision cues do not flip greedy output on GPT-2; cue span does not systematically write toward C.
- [x] Bank-W persistence on this bank requires bank W in the prompt (E8).
- [x] Qwen3.5-4B: forced-W revision succeeds while mid-depth C-swap still shifts preference
  (Δ ≈ −2.1, bootstrap 95% CI [−2.43, −1.69]; all 8 items negative; rand control ≈ 0).

### Ruled out (on these banks)

- [x] Generic “momentum resists the cue.”
- [x] Wrong > correct asymmetry — forced W vs C near-symmetric (`forced_W_vs_C_*`).
- [x] Self-generated bank-W commitment without typing (E8).
- [x] Full behavioral replication of GPT-2 sticking on Qwen (G1 fail).

### Do not claim without new experiments

- [ ] Generic commitment / momentum across tasks or models.
- [ ] That post-training *always* leaves anchors intact — only that revision and causality **can** dissociate.
- [ ] Generality beyond 8 items / one cue family (Qwen larger bank still desirable).
- [ ] E9–E11 / self-generated redesign — **deferred; do not expand before MATS deadline.**

---

## 8. Related work (citations — scoped)

Use these to frame **anchoring** and the **revision-masking** story; not a literature review.

1. **Olsson et al. (2022).** *In-context Learning and Induction Heads.* arXiv:2209.11895.  
   Pattern-completion / copy-like circuits — contrast with E4 integrated residual swap.
2. **Elhage et al. (2021).** *A Mathematical Framework for Transformer Circuits.*  
   https://transformer-circuits.pub/2021/framework/ — induction / composition background.
3. **Belrose et al. (2023).** *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.  
   Meter choice; supports E4b “readable ≠ causal.”
4. **nostalgebraist (2020).** *interpreting GPT: the logit lens.* LessWrong.  
   Secondary meter; E0 motivates not relying on it alone.
5. **Muennighoff et al. (2025).** *s1: Simple test-time scaling.* arXiv:2501.19393.  
   “Wait” / budget forcing → behavioral revision; our Qwen result asks whether the commitment lever vanishes.

---

## 9. Qwen3.5-4B — behavior–causality dissociation (promoted)

**Status:** headline extension, not a table-row under “external validity.”  
**Full prose:** blog §3 · **Session:** [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md)

### 9.1 Behavioral floor

Forced-W: mean score **−3.38** (SEM 0.26) @ answer_pos; **0/8** top-1 = W (**100%** = C).  
Forced-C: impulse still sticks (100% top-1 = C). Wrong-specific revision success.

### 9.2 Causal swap

W_window C-swap L8–23: mean Δ **−2.10** (SEM 0.21); bootstrap 95% CI **[−2.43, −1.69]** (10k resamples).  
Per-item Δ (all negative): −2.76, −2.49, −2.32, −2.31, −2.29, −2.01, −1.76, −0.86.  
Rand-pos control: **−0.02**. GPT-2 ref: −4.94.

### 9.3 Dissociation

Revision succeeds; mid-depth answer-locus state remains a causal lever (~43% of GPT-2 magnitude, >> control).  
**Claim:** Successful behavioral revision can **mask rather than remove** an anchoring mechanism.

---

## 10. Limitations (honest paragraph)

GPT-2 small on short arithmetic is a minimal testbed; both models use n=8 item banks; Qwen required
`answer_pos` scoring for whitespace. Causal interventions identify necessity of W_window state but
do not uniquely identify minimal heads. E9–E11 and larger Qwen banks are deferred. Results
characterize **forced** answer anchoring on GPT-2 and a **scoped** dissociation on Qwen—not a
universal claim about all post-trained models, or that RL specifically caused Qwen’s revision.

---

## 11. Generalization agenda (within Priority A) — do not expand E9–E11 now

| Extension | Adds | Effort / status |
|---|---|---|
| Forced **C** spine | Wrong vs right asymmetry | **Done** — symmetric |
| Self-aligned persistence | Gen scored on own axis | **Done** |
| Qwen3.5-4B dissociation | Behavior vs causality | **Done** — **promoted to §9 / blog hook** |
| Larger Qwen bank (15–20) | Stats robustness | Moderate — optional polish |
| User-message framing | Restore sticking? | Low — good MATS proposal item |
| E9–E11 / self-gen redesign | Scope creep | **Do not touch before deadline** |

---

## 12. Document structure — **blog post** (chosen)

See [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md):

1. Hook — Qwen dissociation (1–2 sentences)
2. Setup
3. GPT-2 mechanism (E1 → E4 flagship → E4b–E8 compressed)
4. **Qwen §** — behavioral floor · causal swap · dissociation
5. Related work (5 citations above)
6. Claims checklist · limitations · one-sentence close

**Conclusion sentence:**  
When a wrong answer is typed into context, GPT-2 builds mid-depth residual state that causally
controls later preference despite revision language; on Qwen3.5-4B the same class of intervention
still moves preference even though the model already revises—suggesting behavioral revision can mask,
rather than erase, an anchoring pathway.

---

## 13. Figures (flagship = E4 + schematic; Qwen optional panel)

| Figure | Source | Role |
|---|---|---|
| **E4 intervention bars** | `E4_outputs/main_interventions.png` | Flagship |
| Schematic | §5 above | Pair with E4 |
| Qwen behavior / causal | `qwen_replication_outputs/*.png` | Dissociation panel |
| Readout vs causality | `E4b_outputs/` | Optional |

---

## 14. Writing workflow (deadline)

1. ~~Choose venue~~ → **blog post** (~2k words) — done.
2. ~~Promote Qwen~~ → §9 + blog §3 — done.
3. ~~Stats on Qwen Δ~~ → SEM + bootstrap CI + per-item list — done.
4. ~~Citations~~ → §8 (5 papers) — done.
5. Polish composite figure (E4 + schematic ± Qwen panel).
6. Paste blog / past-work excerpt into MATS form; keep E9–E11 out.

---

*End of Priority A write-up draft.*

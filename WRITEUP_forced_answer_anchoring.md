# Write-up Draft — Priority A: Forced Answer Anchoring in GPT-2 Small

**Status:** draft for internal use / outline expansion  
**Audience:** you, collaborators, eventual blog or short paper  
**Evidence base:** E0–E8 (+ E4b) on GPT-2 small, 8-item forced-error arithmetic bank  
**Date:** 2026-08-31

Related: [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md), [`REDESIGN_OPTIONS.md`](REDESIGN_OPTIONS.md)

---

## Working title options

1. **How a typed-in wrong answer persists through a revision cue** (descriptive)
2. **Answer-slot anchoring:** causal structure of context-bound arithmetic preferences in GPT-2
3. **Not momentum: mechanistic persistence of a forced answer token in a transformer**

Pick (1) for clarity; (2) for mech-interp framing; (3) if you want the E8 boundary upfront.

---

## Abstract (draft ~150 words)

We study a minimal setting where a wrong arithmetic answer appears in context and a revision cue
fails to change the model's next-token preference. On GPT-2 small, revision cues never flip greedy
output to the correct answer (E1). Using direct logit attribution, tuned-lens readouts, and
residual patching, we map a causal pipeline: late post-embedding writes around the forced-answer
locus build mid-depth residual state that determines preference at a later probe position; attention
routing from the probe to that locus contributes but is not the primary constraint. Operand
corruption softens preference without clean recomputation; the revision span does not write toward
the correct answer. A self-generated vs forced comparison shows bank-specific wrong-answer
persistence requires the token in the prompt—not generic self-commitment on this task bank. We
frame this as **forced answer anchoring**: a locatable, intervenable mechanism for carrying a
context-supplied answer forward, distinct from revision success or model-generated error commitment.

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

- [ ] Typed-in wrong answer at the first answer slot causally influences preference at a later `=`.
- [ ] Effect is localized to answer-locus residual (W_window), strongest mid-depth (L5–L8).
- [ ] Late layers at t* are causally load-bearing despite earlier readable preference.
- [ ] Routing to W_window contributes; content swap is the stronger intervention.
- [ ] Revision cues do not flip greedy output; cue span does not systematically write toward C.
- [ ] Bank-W persistence on this bank requires bank W in the prompt (E8).

### Do not claim without new experiments

- [ ] Generic commitment / momentum across tasks or models.
- [ ] Whether self-generated impulses lack persistence — **ruled out** on aligned axis (`self_aligned_persistence_session_summary.md`).
- [ ] Revision "resistance" — prefer "non-recruitment" or "anchoring."
- [ ] Wrong > correct asymmetry — **ruled out** (forced W vs C near-symmetric; see `forced_W_vs_C_session_summary.md`).
- [ ] Generality beyond 8 items / one cue family.

---

## 8. Related work (positioning bullets)

Use these to frame **anchoring**, not failed momentum:

- **In-context learning / priming** — early context shapes later predictions; we add **causal**
  localization and a failed-revision behavioral template.
- **Sycophancy / user-answer bias** — models echo user-supplied answers; we give a **circuit-level**
  account on a minimal task.
- **Induction / copying heads** — literal token repetition; compare whether W_window effect is
  copying-like vs integrated state (E4 swap suggests integrated residual, not mere embed).
- **"Wait" / self-correction** on reasoning models — different regime (often larger models,
  sometimes successful correction); we document a **failure mode** with mechanism.
- **Logit lens / tuned lens** — E0 motivates meter choice; E4b motivates not trusting early readout.

*Action item:* add 3–5 specific citations when targeting a venue.

---

## 9. Limitations (honest paragraph)

GPT-2 small on short arithmetic is a minimal testbed: greedy continuations from `{a}+{b}=` rarely
match bank W/C; revision never succeeds; depth is only 12 layers. We use a fixed 8-item bank and
one primary cue. Causal interventions identify necessity of W_window state but do not uniquely
identify minimal sufficient heads. J-lens and formation-threshold extensions were deferred. Results
characterize **forced** wrong-answer anchoring; E8 explicitly limits extension to self-generated
errors on this bank.

---

## 10. Generalization agenda (within Priority A)

Extensions that **strengthen** the anchoring paper without changing the core claim:

| Extension | Adds | Effort |
|---|---|---|
| Forced **C** spine | Wrong vs right asymmetry | Low — **done** (`forced_W_vs_C_*`) |
| Self-aligned persistence | Gen scored on own axis | Low — **done** (`self_aligned_persistence_*`) |
| More items / cues | Robustness | Low |
| Second model (e.g. Qwen3.5-4B) | External validity | Medium — **planned:** [`plan_qwen_anchoring_replication.md`](plan_qwen_anchoring_replication.md) |
| User-message framing | "The answer is 25" vs bare `= 25` | Low |
| E9 α-sweep on patch | Nonlinearity of anchor strength | Medium |
| Compare ICL-planted wrong answer | vs user-typed W | Medium |

---

## 11. Suggested document structure (full paper / long post)

1. **Introduction** — provisional answers in context; revision cue puzzle (§1 above)
2. **Setup** — prompt, bank, metrics (§3)
3. **Behavioral baseline** — E1 (+ soft metrics paragraph)
4. **Where is the answer written?** — E2
5. **What does t* read?** — E3
6. **Causal necessity of answer-locus state** — E4 (+ per-layer figure)
7. **Readout vs causality at t*** — E4b
8. **Routing vs content** — E5
9. **Modulation: operands and cue** — E6, E7
10. **Scope: forced vs self-generated** — E8
11. **Discussion** — anchoring frame, related work, limitations
12. **Conclusion** — one sentence (below)

**Conclusion sentence (draft):**  
We show that when a wrong answer token is supplied in context, GPT-2 small builds mid-depth
residual state at that answer locus that causally controls later preferences despite revision
language; this is mechanistically rich answer-slot anchoring, not cue resistance or generic
self-commitment on our task bank.

---

## 12. Figures to assemble from existing outputs

| Figure | Source |
|---|---|
| Behavioral: score + top-1 by condition | E1 curves, E6/E8 bar plots |
| Write-wave heatmap | E2 notebook |
| t* tuned-lens depth curve | E3 outputs |
| E4 intervention bar chart | E4_outputs |
| Readout vs causality overlay | E4b_outputs |
| Path ablation vs content | E5_outputs |
| E8 forced vs self-gen | E8_outputs |

---

## 13. Next steps for *you* (writing workflow)

1. **Choose venue shape:** blog post (~2k words) vs workshop paper (~6–8 pp).
2. **Draft §4** first — results are already frozen in session summaries.
3. **One main figure:** schematic (§5) + E4 bar chart.
4. **Abstract** — refine §Abstract after §4 draft.
5. ~~Optional: run **forced C** (Option 5 in REDESIGN_OPTIONS) for one new paragraph on asymmetry.~~ **Done** — symmetric; paragraph draft in `forced_W_vs_C_session_summary.md`.

---

*End of Priority A write-up draft.*

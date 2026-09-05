# Behavioral revision can leave an anchoring pathway intact

**A short mechanistic story from GPT-2 → Qwen3.5-4B**  
Misrak Berhe · draft for MATS / portfolio · 2026-09-03  
~2,000 words · artifacts in [`github.com/misrakberhe/fluid_dynamics`](https://github.com/misrakberhe/fluid_dynamics)

---

**Claim:** On Qwen3.5-4B, the model can *say* the revised (correct) answer while still having an editable internal copy of the typed wrong answer.

- **Behavior:** after “Wait, let me recompute,” greedy output is C on 8/8 items.
- **Causal check:** overwriting residual activations in the **middle layers**, at the token positions around that typed answer, still shifts preference (Δ ≈ −2.1 vs ≈ 0 at random positions; bootstrap 95% CI [−2.43, −1.69]).

So revision changed the output without removing that pathway. What follows walks back to the GPT-2 mechanism underneath it, then returns to Qwen with the full subsection structure.

**Figure:** [`figures/flagship_ABC.png`](figures/flagship_ABC.png) (plain-language panels + key).  
**Terms:** skim the [glossary](GLOSSARY.md) first if W / C-swap / mid-depth are unfamiliar.

---

## Glossary (short)

| | |
|---|---|
| **W / C** | Wrong vs correct answer we put in the prompt. |
| **t\*** | Probe position: the final `=` where we score preference / greedy top-1. |
| **Mid-depth** | Middle layers of the network (not embedding, not the last layers). GPT-2 peak causal effect ≈ **L5–L8**; Qwen patch band ≈ **L8–23**. |
| **Preference score** | logit(W) − logit(C) at the final `=` (>0 prefers W). C1 plots this. |
| **Δ** | That score after an edit minus before. Negative = shifted toward C. B and C2 plot this. |
| **Stored answer state** | Hidden activations at the typed-answer tokens; overwriting them moves later preference. |
| **W_window** | Token positions around the typed wrong answer (a small span centered on `{W}`), not a layer range. |
| **Edit state** | Replace those activations with a copy from a correct- or wrong-answer run. |
| **Keep state; block writes** | Leave stored activations unchanged; only block new layer writes at that spot. Almost no effect (unlike overwrite). |
| **Anchoring** | A typed-in answer keeps shaping later preference through **stored answer state** at those tokens (not just by sitting in the text). |

Full table: [`GLOSSARY.md`](GLOSSARY.md).

---

## 1. The setting

Provisional answers show up everywhere: user messages, drafts, chain-of-thought. Later tokens may need to continue or revise them. When the prompt says “Wait, let me recompute” and behavior does *or doesn’t* change, is that internal resistance, a cue that never engages correction, or simple propagation of what’s already in context?

Template (forced-answer / forced-error):

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
```

- **W** = typed wrong answer · **C** = paired correct answer  
- Probe **t\*** = final `=` · score = logit(W) − logit(C) · greedy top-1  

Primary model for the mechanism map: **GPT-2 small** (8-item multi-digit bank). External check: **Qwen3.5-4B** (8-item single-digit bank; tokenization forced a dedicated bank).

---

## 2. GPT-2: revision fails, and we can see why

The original motivation was “momentum”: does something commit mid-trajectory so that a later “Wait” doesn't redirect it? The data forced a narrower, more useful claim.

### Behavioral floor (E1)

Across cue phrasings, greedy decode at t\* **never** emits C. Forced-W mean score ≈ **+2.71**; greedy top-1 was the typed wrong answer on **8/8** items. Soft metrics move under cue vs filler, but there is no revision success. The phenomenon is **persistence**, not partial correction.

### Where preference is written and read (E2–E3)

Preference along the W−C direction is added **after the embedding**, across several late layers (attention + MLP) around the typed answer—not as a single mid-layer update, and not mostly from the embedding itself. At t\*, tuned-lens preference emerges mid-depth; OV-weighted attention from the probe puts the most weight on the **token positions around the typed wrong answer** (**W_window**) among the regions we checked.

### Causal core (E4) — flagship result

Mid-depth residual state at the **token positions around the typed wrong answer** (**W_window**) is necessary and sufficient for preference at t\*:

| Intervention | mean Δ score @ t\* |
|---|---|
| C-swap at typed-answer tokens (necessity) | **−4.94** |
| W-patch at typed-answer tokens (sufficiency) | **+4.92** |
| Layers with largest patch effect | **L5–L8** |
| Write ablation at those tokens | ≈ −0.3 (weak) |

Patch the **accumulated residual**, not individual layer writes. Ablating new writes there barely moves the score; swapping the residual collapses or restores it.

**Figure:** `E4_outputs/main_interventions.png` — C-swap and W-patch dominate; write ablations and late-only swaps do not.

```text
[operands] = [W typed] . [revision cue] [operands] = [t*]
                |                              ^
    late-layer residual updates (E2)           |
                v                              |
   typed-answer tokens L5–8 (E4 main site) ---- attn path (E5, weaker)
                |                              |
   preference causally fixed at t* L9–11 (E4b) +
```

### Readout vs causality (E4b)

Tuned-lens readout at t\* can favor W from ~L4; editing t\*’s own residual only strongly affects preference around ~L9–L11. **Readable ≠ necessary** ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112); cf. logit lens, [nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)). Treating early preference as “already decided” would overclaim.

### Constraints in one breath (E5–E7)

Path-blocking attention from t\* to those typed-answer tokens hurts (Δ ≈ −2.3) but less than content swap (−4.9)—routing helps; content matters more. Operand corruption softens preference without clean recomputation of the visible second sum. Activations over the revision text do **not** systematically shift toward C—so “momentum resists the cue” is the wrong story; better: **the cue does not overwrite the stored answer toward C**.

### Scope gate (E8)

On this 8-item set, persistence of the *designated* wrong answer **requires that wrong answer in the prompt**. Self-generated answers stick to *their own* token, not to that designated W. Forced W and forced C are near-symmetric (|W|/|C| ≈ 1.01)—generic answer-slot binding, not a wrong-answer bias.

**Reframe:** this is **forced answer anchoring**—a locatable, intervenable mechanism for carrying a context-supplied answer forward—not generic “momentum” or self-commitment to the same arithmetic error. That honesty is what makes the Qwen dissociation interpretable: we know what mechanism we are probing when we ask whether revision removed it.

---

## 3. Qwen3.5-4B: revision succeeds; the pathway remains

This is not a footnote under “external validity.” It is a distinct claim: **behavioral revision can coexist with a surviving causal anchoring pathway.**

Setup differs only where it must: single-digit items (multi-digit answers split into multiple tokens under Qwen tokenization); score at **answer_pos** after a greedy whitespace prefix (digit is not top-1 at raw t\*); mid-depth band **L8–23** (25–75% of 32 layers). Same C-swap + random-position control protocol.

### 3.1 Behavioral floor — revision is successful

| | GPT-2 @ t\* | Qwen @ answer_pos |
|---|---|---|
| Mean preference score after forced W (logit(W) − logit(C); >0 prefers W) | **+2.71** | **−3.38** (std. error of the mean 0.26) |
| Revision successful? (greedy top-1 = C after forced W) | **No** (top-1 = W on 8/8) | **Yes** (top-1 = C on 8/8) |
| Forced C: greedy top-1 = C | 100% | 100% |

On forced-W prompts, Qwen **revises to C** on every item. Forced-C still sticks—so this is **wrong-specific revision success**, not failure to read the slot.

### 3.2 Causal swap — mid-depth state still moves preference

| Intervention | mean Δ (preference score) | Std. error of the mean | bootstrap 95% CI |
|---|---|---|---|
| C-swap at typed-answer tokens (L8–23) | **−2.10** | 0.21 | **[−2.43, −1.69]** |
| Random-position control | **−0.02** | 0.04 | — |
| GPT-2 reference (L5–11) | −4.94 | — | — |

Per-item C-swap Δs (all negative; sorted):  
−2.76, −2.49, −2.32, −2.31, −2.29, −2.01, −1.76, **−0.86**.

The mean is not driven by two outliers. Seven of eight items sit between −1.76 and −2.76; one weaker item (1+6) is still negative and directionally consistent. Off-locus control is null. Effect size is ~43% of GPT-2’s, but clearly separated from control.

**Figures:** `qwen_replication_outputs/behavior_gpt2_vs_qwen.png`, `replication_causal_gpt2_vs_qwen.png`.

### 3.3 The dissociation

| | Behavioral sticking (forced W) | Causal C-swap at typed-answer tokens |
|---|---|---|
| **GPT-2 small** | Yes (+2.7, 100% W) | Strong (Δ ≈ −4.9) |
| **Qwen3.5-4B** | No (revises to C) | Present (Δ ≈ −2.1 vs ≈ 0 control) |

GPT-2 shows **both** sticking and strong patching. Qwen shows **revision success** with a **non-null causal localization**—mid-depth state at the typed-answer tokens still modulates digit preference even when greedy output already favors C. The swap does not need to flip top-1 to be informative: baseline already favors C, and C-swap pushes further toward C relative to an off-locus control that does nothing.

**Publishable claim (scoped):** On this item set, successful behavioral revision can **change the output** without **removing** the editable anchoring pathway. Surface correction is not evidence that the underlying mechanism is gone.

That is the thread from “Wait” tokens and test-time revision ([Muennighoff et al., 2025, *s1*](https://arxiv.org/abs/2501.19393)): budget-forcing with “Wait” boosts revision *behavior*. Our causal check asks whether the commitment mechanism actually disappears. On Qwen3.5-4B in this setting: **not fully**.

Why this matters for alignment-adjacent work: evaluations that only check whether the model *says* the revised answer can miss intact editable pathways. A model that revises on the surface while remaining patchable at the typed-answer tokens is a different safety object than one whose anchoring pathway has been extinguished—or never formed.

---

## 4. Related work (tight)

**Anchoring vs copying.** Induction heads implement pattern completion of the form [A][B] … [A] → [B] ([Olsson et al., 2022](https://arxiv.org/abs/2209.11895); [Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/)). Our E4 residual swap moves integrated mid-depth *state*, not mere embedding at the typed token—closer to an answer-encoding residual than literal copy-head behavior, though a head-level decomposition is future work.

**Readout tools.** Early readout can mislead: we follow tuned-lens practice ([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)) and treat E4b’s readout–causality gap as a methodological constraint, not a curiosity.

**Revision / “Wait”.** *s1* shows that appending “Wait” under budget forcing can induce self-correction and scale test-time compute ([Muennighoff et al., 2025](https://arxiv.org/abs/2501.19393)). We study the complementary failure and partial-success modes: when revision language fails (GPT-2) vs succeeds while an editable anchoring pathway remains (Qwen).

---

## 5. Claims checklist

### Safe to claim

- Typed-in answer-slot residual on GPT-2 is causally necessary/sufficient for later W−C preference (E4, Δ ≈ ±5, L5–L8).
- Readout can favor W before preference is causally fixed at t\* (E4b).
- Revision cues fail behaviorally on GPT-2 and do not shift activations toward C (E1, E7).
- On this GPT-2 item set, persistence of the designated wrong answer requires W in the prompt (E8); forced W/C anchoring is symmetric.
- On Qwen3.5-4B (n=8 single-digit), forced-W revision succeeds (0/8 top-1 = W) while C-swap at the typed-answer tokens still shifts preference (mean Δ −2.10, 95% bootstrap CI [−2.43, −1.69], all items negative; rand control ≈ 0).

### Ruled out (on these item sets)

- Generic “momentum resists the cue.”
- Wrong > correct anchoring asymmetry (GPT-2).
- Self-generated commitment to the designated wrong answer without typing it (E8).
- Full behavioral replication of GPT-2 sticking on Qwen.

### Needs new experiments (deferred—do not dilute now)

- Larger Qwen item set (15–20 items) / per-layer sweep / prompt framing that restores sticking.
- Head-level identity of the write/read pathway on Qwen.
- Follow-on formation / self-generated redesign experiments (E9–E11).

---

## 6. Limitations

Eight items per model; GPT-2 arithmetic is a minimal testbed; Qwen scoring required an `answer_pos` fix for whitespace. Causal localization does not uniquely identify minimal heads. We do **not** claim this dissociation holds for all post-trained models—only that in this matched protocol, revision success and causal patchability **dissociate**. We also do not isolate RL vs instruction tuning as the cause of Qwen’s revision behavior.

---

## 7. One sentence

**When a wrong answer is typed into context, GPT-2 builds mid-depth residual state that causally controls later preference despite revision language; on Qwen3.5-4B the same class of intervention still moves preference even though the model already revises—so behavioral revision can leave an anchoring pathway intact rather than erase it.**

---

### Reproduce

- GPT-2 E4: `python E4_content_patching.py` · figure `E4_outputs/main_interventions.png`  
- Qwen: `python qwen_anchoring_replication.py phase4` · see `qwen_anchoring_replication_session_summary.md`  
- Spine one-pager: [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md)

### Flagship figure for applications

**[`figures/flagship_ABC.png`](figures/flagship_ABC.pdf)** — panels A (pipeline schematic) · B (E4 necessity/sufficiency) · C (GPT-2 vs Qwen behavior + causal dissociation).  
Regenerate: `python make_flagship_figure.py`

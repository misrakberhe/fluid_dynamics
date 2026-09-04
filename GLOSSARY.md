# Glossary — Forced Answer Anchoring

Plain-language definitions for the blog, figure, and MATS application.  
Jargon in parentheses is what appears in code / session summaries.

---

## The setup

| Term | Meaning |
|---|---|
| **W** | The **wrong** answer we type into the prompt (e.g. `25` for 12+15). |
| **C** | The **correct** paired answer (e.g. `27`). |
| **Forced W / forced C** | Prompt where that answer is written in the first `=` slot before “Wait, recompute…”. |
| **Revision cue** | Text like “Wait, let me recompute” that *asks* the model to try again. |
| **Probe / final `=`** | The second equals sign — where we measure what the model prefers next. (Also called **t\***.) |

Prompt skeleton:

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
                                              ↑ probe
```

---

## What we measure

| Term | Meaning |
|---|---|
| **Preference score** | logit(W) − logit(C) at the probe. **> 0** = prefers wrong; **< 0** = prefers correct. C1 plots this. |
| **Top-1** | The single most likely next token (greedy output). |
| **Δ (delta)** | Score after the edit minus before. Negative Δ = moved toward C. B and C2 plot this — not a probability. |

---

## Inside the model

| Term | Meaning |
|---|---|
| **Hidden / residual state** | The model’s internal activations at a token and layer — not the text itself. |
| **Answer locus / typed-answer tokens** | Token positions of the first answer (where W or C was written). |
| **W_window** | Token positions around the typed wrong answer (small span centered on `{W}`), not a layer range. Code name for that locus when patching. |
| **Stored answer state** | Hidden activations at the typed-answer tokens (mid-depth). Editing them moves later preference. |
| **Overwrite state** | Replace those activations with a copy from a correct-run or wrong-run (C-swap / W-patch). |
| **Keep old state; block new writing** | Write ablation: don’t replace the residual — only zero new attn/MLP writes. Preference barely moves. |
| **Late-layer residual updates** | Several late layers add W-vs-C preference into that region (attention + MLP), not just the embedding. |
| **Mid-depth (L5–L8 on GPT-2)** | Layers where that stored state is most causally important. |

---

## Interventions (Panel B / C2)

| Plain label | Technical name | What we do |
|---|---|---|
| **Replace with correct-run state** | C-swap / necessity | Overwrite answer-token activations with those from a run that had C typed in. |
| **Insert wrong-run state** | W-patch / sufficiency | Paste answer-token activations from a W-run into a C-run. |
| **Zero new writes only** | Write ablation | Kill that layer’s new contribution but leave earlier stored state. |
| **Edit elsewhere** | Random-position control | Same swap at unrelated tokens — should do ~nothing if the effect is localized. |

---

## The punchline words

| Term | Meaning |
|---|---|
| **Anchoring** | Context-supplied answer keeps shaping later predictions via locatable internal state. |
| **Behavioral revision** | Greedy output switches to the correct answer after the cue (Qwen on forced W). |
| **Pathway remains** | Editing the stored answer-state still moves preference, even if output already looks revised. |
| **Dissociation** | Behavior and causal structure disagree (Qwen revises; swap still works). |

---

## Experiment IDs (optional)

| ID | One line |
|---|---|
| E1 | Revision cues never flip GPT-2 output to C. |
| E2 | Preference added across late layers at the typed-answer tokens. |
| E3 | Probe attends to / reads from that locus. |
| E4 | Stored state there is necessary & sufficient (the ±5 result). |
| E4b | Readable early ≠ causally fixed early. |
| E5 | Attention path helps; content swap matters more. |
| E6 | Changing visible operands softens but doesn’t cleanly recompute. |
| E7 | Cue span doesn’t write toward C. |
| E8 | Designated-W sticking needs W in the prompt on this item set. |

# MATS Application — Document Structure & Source Map

**Purpose:** Fill-in template for the MATS application (past work + summer proposal).  
**Status:** outline only — draft prose in each `[WRITE]` block.  
**Date:** 2026-08-31  
**Related:** [`MATS_application_boosters.md`](MATS_application_boosters.md), [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md), [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md)

---

## Before you write

| Item | Your value |
|---|---|
| MATS past-work word limit | ___ words |
| MATS proposal word limit | ___ words |
| Target mentor / lab angle | mech interp / alignment-adjacent anchoring / ___ |
| Primary track | **A — Generalize anchoring** (recommended) |
| Repo URL to link | `github.com/misrakberhe/fluid_dynamics` |

**Framing rules** (from [`MATS_application_boosters.md`](MATS_application_boosters.md) § Framing checklist):

- Lead with **forced answer anchoring**, not "momentum in transformers."
- Center **E4 residual patching** in the first 3 sentences of past work.
- E1/E7: revision **fails** because the cue **doesn't recruit** correction — not "model resists."
- E8 scope in one sentence: bank-W persistence **requires W in the prompt** on this bank.
- Include the pivot: hypothesis → test → revise (momentum → anchoring → E8 scope → Qwen dissociation).

---

## Attachments checklist

| Attachment | File | Status |
|---|---|---|
| Main causal figure | `E4_outputs/main_interventions.png` | [ ] attach |
| Mechanism schematic | `WRITEUP_forced_answer_anchoring.md` §5 (redraw or export) | [ ] create |
| Optional: readout vs causality | `E4b_outputs/readout_vs_causality_raw.png` | [ ] attach |
| Optional: Qwen comparison | `qwen_replication_outputs/replication_causal_gpt2_vs_qwen.png` | [ ] attach |
| Repo link | `E4_content_patching.py`, session summaries | [ ] link |

**Highest ROI gap:** one polished composite figure (E4 bar + schematic). See [`MATS_application_boosters.md`](MATS_application_boosters.md) §1.

---

# PART 1 — Past research (~200–400 words)

## 1. Title / one-line hook (~15 words)

**Source:** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) § Working title options

**Pick one:**

- [ ] *How a typed-in wrong answer persists through a revision cue* (descriptive)
- [ ] *Answer-slot anchoring: causal structure of context-bound preferences in GPT-2* (mech interp)
- [ ] *Not momentum: mechanistic persistence of a forced answer token* (pivot upfront)

`[WRITE]` One sentence: what you found and why it matters mechanistically.

---

## 2. Setup & question (~40–60 words)

**Sources:**

- [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) § The question
- [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) §1.1 Motivation

**Include:**

- Prompt skeleton: `{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =`
- Probe `t*` = final `=`; score = logit(W) − logit(C)
- Model: GPT-2 small; 8-item forced-error bank
- Question: does a context-supplied answer **persist** through a revision cue, and **how**?

`[WRITE]` ___

---

## 3. Behavioral baseline (~30–40 words)

**Sources:**

- [`E1_session_summary.md`](E1_session_summary.md)
- [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) — E1 row

**Key numbers:**

- Revision cues **never** flip greedy output to C (0% success on bank)
- Forced-W mean score @ t*: **+2.71**; top-1 = bank W **100%**

`[WRITE]` State that behavior sticks; do **not** frame as "momentum resisting."

---

## 4. Causal core — E4 (~60–80 words) ⭐ lead here if trimming

**Sources:**

- [`E4_session_summary.md`](E4_session_summary.md) §2–3
- [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) §4.4
- Data: `E4_outputs/intervention_summary.csv`, `E4_outputs/e4_verdict.json`
- Figure: `E4_outputs/main_interventions.png`

**Key numbers:**

| Intervention | mean Δ score @ t* |
|---|---|
| W_window C-swap (necessity) | **−4.94** |
| W_window W-patch (sufficiency) | **+4.92** |
| Per-layer peak leverage | **L5–L8** |
| Write ablation @ W_window | ≈ −0.3 (weak) |

**Claim:** Mid-depth residual state at the answer locus (`W_window`) is **necessary and sufficient** for preference at `t*`.

`[WRITE]` ___

**Paste-ready shortcut** (edit as needed):

> Using residual patching, I showed mid-depth state at the answer locus (W_window, layers 5–8) is necessary and sufficient for later preference at the final `=` (Δ≈±5, E4).

---

## 5. Readout vs causality — E4b (~25–35 words)

**Sources:**

- [`E4b_session_summary.md`](E4b_session_summary.md)
- [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) §4.5
- Figure: `E4b_outputs/readout_vs_causality_raw.png`

**Key numbers:**

- Tuned-lens readout favors W from ~**L4**
- Causal fixation at t* ~**L9–L11**
- **Readable ≠ necessary**

`[WRITE]` One sentence on meter discipline.

---

## 6. Constraints — E5, E6, E7 (~50–70 words, compress if needed)

**Sources:**

| Exp | Summary | Key Δ / result |
|---|---|---|
| E5 | [`E5_session_summary.md`](E5_session_summary.md) | Path block t*→W_window **−2.3** vs content swap **−4.9** |
| E6 | [`E6_session_summary.md`](E6_session_summary.md) | Operand corrupt **−2.3**; no clean recompute |
| E7 | [`E7_session_summary.md`](E7_session_summary.md) | Cue does **not** write toward C |

**Sources (prose):** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) §4.6–4.8

`[WRITE]` Routing helps but content dominates; operands modulate; cue not recruited for correction.

---

## 7. Scope & reframe — E8 (~40–50 words)

**Sources:**

- [`E8_session_summary.md`](E8_session_summary.md)
- [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) § What E8 changed

**Key numbers:**

| Condition | mean W−C @ t* | top-1 |
|---|---|---|
| Forced W in prompt | +2.71 | bank W 100% |
| Self-generated | +0.57 | own token 100% |

**Claim:** E2–E7 describe **typed-in** answer persistence, not generic self-commitment. Reframe as **forced answer anchoring**.

`[WRITE]` ___

---

## 8. Extensions (optional paragraph if word budget allows, ~60–80 words)

### 8a. Forced W vs C asymmetry — done

**Source:** [`forced_W_vs_C_session_summary.md`](forced_W_vs_C_session_summary.md)  
**Data:** `forced_W_vs_C_outputs/summary.csv`

- Forced W: +2.71, 100% top-1 = W; forced C: −2.68, 100% top-1 = C
- Asymmetry ratio ≈ **1.01**; causal swaps mirrored (Δ ≈ ±5)
- **Wrong and correct anchoring are symmetric** — generic slot binding, not wrong-answer bias

`[WRITE]` ___

### 8b. Qwen external validity — done

**Source:** [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md)  
**Data:** `qwen_replication_outputs/verdict.json`  
**Figures:** `behavior_gpt2_vs_qwen.png`, `replication_causal_gpt2_vs_qwen.png`

- G1 behavior **fail**: forced-W revises to C (0% top-1 = W; mean −3.4)
- G2 causal **pass**: C-swap Δ ≈ **−2.1** vs rand control ≈ **0** (GPT-2: −4.9)
- **Behavior–causality dissociation**

**Paste-ready:**

> Qwen3.5-4B does not behaviorally anchor forced wrong answers (revision succeeds), but mid-depth impulse-window C-swap still shifts answer-digit preference (Δ ≈ −2.1 vs ≈ 0 control) — causal localization without GPT-2-style sticking.

`[WRITE]` ___

---

## 9. Artifacts & reproducibility (~20–30 words)

**Sources:**

- Code: `E4_content_patching.py`, `qwen_anchoring_replication.py`
- Narrative: [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md)
- Full trail: `E{N}_session_summary.md` for N = 0–8 (+ E4b)

`[WRITE]` Link repo; mention session summaries and runnable E4 script.

---

## 10. Past-work assembly (full draft)

**Source:** [`MATS_application_boosters.md`](MATS_application_boosters.md) § Paste-ready paragraph

Combine sections 2–9. Target: ___ / ___ words.

`[PASTE FULL DRAFT HERE]`

---

# PART 2 — Research proposal (~300–500 words)

**Recommended track:** **A — Generalize anchoring** (forced C and Qwen already done; propose what comes next).

**Sources for fork options:** [`REDESIGN_OPTIONS.md`](REDESIGN_OPTIONS.md), [`MATS_application_boosters.md`](MATS_application_boosters.md) § Research proposal

---

## 1. Motivation & gap (~60–80 words)

**Build on:**

- GPT-2 anchoring is mapped; scope bounded (E8)
- Qwen shows **dissociation** — causal lever survives when behavior revises
- Open: **when** does anchoring break vs persist? **what** determines revision success?

`[WRITE]` Why this is a good MATS summer project; connect to mentor interests (mech interp / sycophancy / context binding).

---

## 2. Research question (~30–40 words)

**Example (Track A):**

> Under what conditions does mid-depth answer-locus residual state causally control final preferences — and when does behavioral revision succeed despite that state?

`[WRITE]` Your precise question: ___

---

## 3. Approach (~80–120 words)

**Three pillars** (adjust to word limit):

| Pillar | What | Builds on |
|---|---|---|
| **A. Prompt framing** | User-message vs bare arithmetic (`"The answer is {W}"` vs `={W}`) | [`REDESIGN_OPTIONS.md`](REDESIGN_OPTIONS.md), plan stretch G6 |
| **B. Revision-success domains** | Find model/task where cues sometimes work; patch success vs failure | Track B in boosters |
| **C. Layer / mechanism refinement** | Per-layer sweep on Qwen; compare write-wave to GPT-2 E2 | Qwen plan stretch G5; `E2_session_summary.md` |

**Methods:** Same protocol — behavioral score + top-1 @ probe; W_window residual C-swap + rand control; TransformerLens patching.

`[WRITE]` ___

---

## 4. Milestones (~80–100 words)

**Source:** [`MATS_application_boosters.md`](MATS_application_boosters.md) § Milestones (adapt — forced C and Qwen done)

| Week | Deliverable |
|---|---|
| 1–2 | Prompt framing variants (user message vs bare `={W}`) on GPT-2 + Qwen |
| 3–5 | Per-layer causal sweep on Qwen; compare leverage depth to GPT-2 L5–8 |
| 6–8 | Revision-success pilot: new domain or model where cues sometimes flip output |
| 9–10 | Write-up with 3 figures; behavior vs causality dissociation as central theme |
| 11–12 | Optional: patch magnitude sweep (E9 α); ICL-planted vs user-typed W |

`[WRITE]` Customize milestones for your target mentor.

---

## 5. Expected outcomes & risks (~50–70 words)

**Possible outcomes:**

- Framing that restores behavioral sticking on capable models
- Clear map of when causal localization exists without behavioral anchoring
- Negative results documented honestly (G3-style)

**Risks:** Tokenization breaks bank; null patches on some models; revision success everywhere → pivot to mechanism of *successful* revision.

`[WRITE]` ___

---

## 6. Why you (~40–60 words)

**Draw from:**

- E0–E8 full spine with session summaries
- Post-spine extensions: forced C, self-aligned persistence, Qwen replication
- Pivot narrative: momentum hypothesis → anchoring reframe → external validity

`[WRITE]` What skills and artifact trail you bring.

---

## 7. Proposal assembly (full draft)

Target: ___ / ___ words.

`[PASTE FULL DRAFT HERE]`

---

# PART 3 — Optional short fields

## Elevator pitch (~50 words)

`[WRITE]` ___

## Reviewer one-liner

**Source:** [`MATS_application_boosters.md`](MATS_application_boosters.md) § Compelling vs solid

> Applicant found a causal bottleneck, bounded the claim honestly, and has a concrete plan to generalize or explain revision failure mechanistically.

`[WRITE]` Customize: ___

## Related work (1–2 sentences, if form asks)

**Source:** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) §2 (if present) or draft from anchoring / ICL / sycophancy angles

`[WRITE]` ___

---

# Quick reference — numbers at a glance

| Metric | GPT-2 | Qwen3.5-4B |
|---|---|---|
| forced-W mean score | +2.71 @ t* | −3.38 @ answer_pos |
| forced-W top-1 = impulse | 100% W | 0% W (100% C) |
| W_window C-swap Δ | −4.94 (L5–11) | −2.10 (L8–23) |
| Rand-pos control Δ | — | −0.02 |
| forced C mean score | −2.68 | −5.50 |
| E8 forced vs self-gen | +2.71 vs +0.57 | — |
| Forced W/C asymmetry | 1.01 | wrong-specific revision |

**Primary data files:** `E4_outputs/intervention_summary.csv`, `forced_W_vs_C_outputs/summary.csv`, `qwen_replication_outputs/verdict.json`

---

# Writing workflow

1. Fill `[WRITE]` blocks in **Part 1 §4 (E4)** and **§7 (E8)** first — highest signal.
2. Add **§8b (Qwen)** if word budget allows — differentiates your application.
3. Assemble **§10**; trim §6 or §8a if over limit.
4. Draft **Part 2** from Track A milestones; personalize for mentor.
5. Create **composite figure** (E4 + schematic) before final submit.
6. Run framing checklist at top of this doc.

---

# File index (all sources)

| Category | Files |
|---|---|
| **Spine narrative** | `E0-E8_spine_narrative.md` |
| **Long write-up draft** | `WRITEUP_forced_answer_anchoring.md` |
| **Application strategy** | `MATS_application_boosters.md` |
| **Session summaries** | `E0_session_summary.md` … `E8_session_summary.md`, `E4b_session_summary.md` |
| **Post-spine** | `forced_W_vs_C_session_summary.md`, `self_aligned_persistence_session_summary.md`, `qwen_anchoring_replication_session_summary.md` |
| **Code** | `E4_content_patching.py`, `qwen_anchoring_replication.py` |
| **Figures** | `E4_outputs/main_interventions.png`, `E4b_outputs/readout_vs_causality_raw.png`, `qwen_replication_outputs/*.png` |
| **Data** | `E4_outputs/`, `forced_W_vs_C_outputs/`, `qwen_replication_outputs/` |
| **Future work** | `REDESIGN_OPTIONS.md`, `plan_qwen_anchoring_replication.md` |

---

*End of MATS application outline.*

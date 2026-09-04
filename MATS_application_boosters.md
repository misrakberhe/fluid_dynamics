# MATS Application — From Solid to Compelling

**Context:** E0–E8 spine on GPT-2 small; core result = **forced answer anchoring** (causal persistence of a typed-in answer token), not generic momentum.  
**Date:** 2026-08-31  
**Related:** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md), [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md), [`REDESIGN_OPTIONS.md`](REDESIGN_OPTIONS.md)

---

## What's already "solid" (don't re-litigate)

You have enough for a **strong past-work** section if you lead correctly:

- **E4:** W_window residual swap Δ ≈ ±5 — causal necessity/sufficiency
- **E4b:** readout emerges ~L4; causal fixation ~L9–11 — meter discipline
- **E5:** content > routing (patch −4.9 vs path block −2.3)
- **E8:** scope boundary — bank W requires typing; intellectual honesty
- **Full artifact trail:** code, notebooks, session summaries, write-up draft

**Lead with E4/E4b, not** "model repeats context."

---

## What moves solid → compelling

You don't need all four. **One** polished addition + tight framing is often enough.

### 1. One application-quality figure (highest ROI)

**What:** Single figure for the app / portfolio:

- Panel A: E4 intervention bar chart (C-swap / W-patch Δ at t*)
- Panel B: schematic from `WRITEUP_forced_answer_anchoring.md` §5 (write → W_window L5–8 → t* L9–11)

**Why:** Reviewers skim; one causal figure beats ten session summaries.

**Source files:** `E4_outputs/`, `E4b_outputs/readout_vs_causality_raw.png` (optional third panel)

**Effort:** ~2–4 hours (matplotlib polish + export PNG/PDF)

---

### 2. Forced C spine (cheap new science)

**What:** Re-run the **behavioral + E4 necessity** pipeline with **C in the impulse slot** instead of W:

- `12 + 15 = 27. Wait…` (C-prompt as primary, not only counterfactual for patching)
- Compare: does correct answer persist equally? Is wrong-W anchoring **asymmetric**?

**Why:** One new paragraph: "wrong vs right anchoring" — shows you extend, not only document.

**Effort:** Low — `build_C_prompt` already exists in `E4_content_patching.py`; mainly aggregate + 1 plot.

**MATS line:** "I tested asymmetry between forced wrong vs forced correct anchoring."

---

### 3. Second model (external validity) — **done (Qwen3.5-4B)**

**What:** Same protocol on one additional model — **completed** on Qwen3.5-4B (see [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md)).

**Result:** G1 behavior **fail** (forced-W revises to C; 0% top-1 = W). G2 causal **pass** (W_window C-swap Δ ≈ −2.1 vs rand control ≈ 0; GPT-2 ref −4.9). Behavior–causality **dissociation**.

**Why it matters:** Answers "GPT-2 only" partially — causal localization survives on a capable model even when behavioral sticking breaks.

**MATS line (paste-ready):**

> Qwen3.5-4B revises forced wrong answers to the correct digit (0/8 stick), but mid-depth impulse-window C-swap still shifts preference (Δ ≈ −2.1, 95% bootstrap CI [−2.43, −1.69]; rand control ≈ 0) — revision can mask rather than erase the causal lever.

**Blog:** [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md)

---

### 4. Short public artifact

**What:** Any one of:

- GitHub README with spine narrative + 2 figures + "how to reproduce E4"
- ~1,500-word blog post from `WRITEUP_forced_answer_anchoring.md` §4
- Colab or notebook link that runs E4 on one item

**Why:** Shows communication + reproducibility; gives reviewers something to click.

**Effort:** Low–medium (mostly editing existing material)

---

## Framing checklist (application text)

### Do

- [ ] Open with **Qwen dissociation** (revision succeeds; causal lever remains), then GPT-2 mechanism
- [ ] Title the work **forced answer anchoring** / **revision can mask anchoring**
- [ ] Center **E4 residual patching** as the GPT-2 causal core
- [ ] Report Qwen Δ with **SEM / CI / per-item** (not a bare mean)
- [ ] State **E8 scope** in one sentence: "bank-W persistence requires W in the prompt on this bank"
- [ ] Frame E1/E7 as: revision **fails** because cue **doesn't recruit** correction, not "momentum resists"
- [ ] Propose a **specific** MATS summer project (Track A)
- [ ] **Do not** expand E9–E11 before deadline

### Don't

- [ ] Lead with "we studied momentum in transformers"
- [ ] Bury Qwen as "Medium effort — done" under external validity
- [ ] Claim self-generated wrong-answer commitment (E8 ruled out on this bank)
- [ ] Oversell GPT-2 arithmetic as "reasoning"
- [ ] Hide the pivot — **hypothesis → test → revise** is a strength for MATS
- [ ] Scope-creep into incomplete E9–E11 / redesign banks

---

## Suggested application structure

### Past research (~200–400 words)

1. Question: does a wrong answer in context persist through a revision cue, and **how** (mechanism)?
2. Behavior: cues never flip output (E1)
3. **Causal core:** W_window L5–8 determines t* (E4); late fixation at t* (E4b)
4. Constraints: routing helps (E5); operands modulate (E6); cue doesn't write toward C (E7)
5. Scope: typed W required for bank-W effect (E8) → reframed as anchoring
6. Link to artifacts (repo / figures)

**Paste-ready paragraph** (customize):

> I built a controlled revision-failure setting on GPT-2 small: `{a}+{b}={W}. Wait, let me recompute. {a}+{b}=`. Revision cues never flip greedy output to the correct answer (E1). Using residual patching, I showed mid-depth state at the answer locus (W_window, layers 5–8) is necessary and sufficient for later preference at the final `=` (Δ≈±5, E4). Early readout at the probe can favor the wrong answer before it is causally fixed late (E4b). Attention routing contributes but content dominates (E5). A forced-vs-self-generated comparison showed bank-specific persistence requires the token in the prompt (E8), so I reframed the result as mechanistic answer-slot anchoring rather than generic commitment. Code, notebooks, and session summaries document E0–E8.

### Research proposal for MATS (~300–500 words)

Pick **one primary track** (+ optional stretch):

| Track | Summer goal | Builds on |
|---|---|---|
| **A — Generalize anchoring** | Second model + forced C asymmetry + user-framing variants | Past work directly |
| **B — Revision success** | Domain/model where cues sometimes work; patch success vs failure | Reopens original motivation |
| **C — Self-aligned binding** | E8b: persistence on gen token + patch at gen locus | Bridges anchoring ↔ commitment |

**Recommendation for mech-interp mentors:** Track A (low risk, clear deliverables).  
**Recommendation if original "revision" story matters:** Track B with Track A week 1–2 as baseline.

---

## Compelling vs solid — reviewer one-liner

| Tier | What they take away |
|---|---|
| **Solid** | "Applicant can run patching and document a spine." |
| **Compelling** | "Applicant found a causal bottleneck, bounded the claim honestly, and has a concrete plan to generalize or explain revision failure mechanistically." |

---

## Priority order (if time-limited) — updated 2026-09-03

1. ~~**Promote Qwen dissociation**~~ — **done** ([`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md); WRITEUP §9)
2. ~~**Qwen stats (SEM / bootstrap CI / per-item)**~~ — **done** (CI [−2.43, −1.69]; all 8 Δ < 0)
3. ~~**Polished E4 + schematic (± Qwen panel)**~~ — **done** ([`figures/flagship_ABC.png`](figures/flagship_ABC.png); `make_flagship_figure.py`)
4. **Paste blog / past-work into MATS form** + Track A proposal
5. **Do not** open E9–E11 or self-generated redesign before deadline
6. Optional: enlarge Qwen bank to 15–20 items if hours remain

---

## Milestones for a 10–12 week MATS proposal (Track A example)

| Week | Deliverable |
|---|---|
| 1–2 | Forced C + forced W comparison (behavior + one patch) |
| 3–5 | Replicate E4 necessity on Pythia (or chosen model) |
| 6–8 | Prompt framing variants (user message vs bare arithmetic) |
| 9–10 | Write-up / blog / short report with 3 figures |
| 11–12 | Optional E9 α-sweep on patch magnitude |

---

## Files to attach or link in application

- [`MATS_application_outline.md`](MATS_application_outline.md) — **fill-in doc structure** (section map + source files)
- [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md) — narrative draft
- [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) — one-page summary
- [`qwen_anchoring_replication_session_summary.md`](qwen_anchoring_replication_session_summary.md) — Qwen external validity
- `E4_outputs/` — key causal numbers
- `qwen_replication_outputs/` — Qwen behavior + causal figures
- `E4_content_patching.py` — reproducibility
- **New:** polished figure (when done)
- **New:** public README or post (when done)

---

## Open questions to resolve

- [ ] MATS word limits for past work vs proposal
- [ ] Which mentor/lab angle (pure mech interp vs alignment-adjacent anchoring/sycophancy)
- [ ] Track A vs B for summer proposal
- [ ] Whether to run forced C before submitting — **done** (`forced_W_vs_C_*`)

---

## Key framing reminders (from discussion)

**Predetermined output (qualified):** On this bank, t* preference is **largely determined by typed answer-slot state** (E4), not fresh recomputation — but revision cue doesn't override (E1/E7), and operands modulate (E6). Say **context-determined / causally anchored**, not "fated before forward pass."

**Anchoring vs momentum:** Stand alone anchoring for past work; momentum is a **fork** needing redesign (revision-success domain or E8-pass bank), not a natural sequel on current results.

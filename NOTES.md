# fluid_dynamics — research notes

Carried from prior chat. Two tracks: near-term mech interp (Group 1), long-term generative model (Group 2).

## Background

- Cog sci (York); senior data platform engineer (Canadian Tire)
- Applied MATS (not accepted); math workable but not strong
- ARENA / Karpathy material largely covered
- **Primary path:** mechanistic interpretability
- **Fallback/parallel:** AI + education (Anthropic Education Labs–style) — not primary
- Interp is prioritized because it builds toward the Group 2 research question

---

## Group 1: Scoped near-term plan (mech interp)

### Test 1 — ready to execute (not yet run)

**Question:** Is there an identifiable point during LLM generation where the model's internal trajectory becomes resistant to revision — i.e., does "momentum" build the deeper into a reasoning chain the model goes, such that a revision cue ("wait, let me reconsider") can only sometimes actually redirect it?

- **Sub-question 1 (binary, per-example):** Does a revision cue correspond to real internal change, or is it surface-only language?
- **Sub-question 2 (momentum test):** Does correction/patching success rate drop off as the revision point occurs later in the trajectory?

**Setup:**

```text
pip install transformer_lens torch
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")  # CPU-fine, ~500MB
# Sanity: model.generate("The capital of France is", max_new_tokens=10)
```

**Method:**

1. Build ~10 simple arithmetic/logic prompts where GPT-2 small errs, followed by a natural revision phrase (e.g. `12 + 15 = 25. Wait, let me recompute. 12 + 15 =`).
2. Use `model.run_with_cache()` for logits/activations at each token.
3. Compare logit distribution for the answer token before vs. after the revision cue — did the predicted top token actually change?
4. Vary how deep into the trajectory the revision cue appears; check whether correction success declines with depth.

**Next step:** build/debug the experiment code step by step.

---

## Group 2: Ni/Se vortex-ring metaphor (long-term, generative/interpretive)

Not yet operationalized. Rich, internally consistent model for thinking — **not** yet falsifiable/measurable. Reframing into measurable claims is PhD-thesis-shaped, not a near-term project.

### Structural model so far

- **Ni** modeled as a torus / vortex-ring; **Se-pulses** pass through and generate it.
- **Formation:** an Se-pulse's *edge* (not interior) experiences shear against other Se-pulses (relative velocity difference is enough; no stationary reference) and rolls up into the core. Interior is carried along but is not the structuring element.
- **Core** = concentrated rotating structure (not the "smoke"/tracer). Candidate physical stand-in for what Ni "is" once formed.
- **Propagation speed vs. core size** trade-off → hypothesized differentiator between primary-Ni and primary-Se styles.
- **Viscous decay** (rings dissipate without re-energizing) → attributed to Se.
- **Leapfrogging** (two similar-strength rings alternate lead indefinitely without merging) — one of two most interesting behaviors.
  - Prediction: only between "peer-weight" insights, not strong conviction vs. stray thought → some intuition-pairs are structurally non-resolving rather than pending resolution.
- **Reconnection** (cores briefly merge/break/reform, sometimes throwing off smaller derivative rings) — the other most interesting behavior.
  - Prediction: should leave a distinct "byproduct" insight at the collision point, separate from either parent — not in standard Jungian vocabulary (contrast with "synthesis").
- **"Meaning"** reframed as **invariance** — whatever stays stable/recognizable across interactions (leapfrogging, reconnection) as stand-in for "holding meaning."

### Open tension

Ni/Se feel introspectively separable, but may not be structurally separate — possibly a regime-within-continuum (vortex core within fluid) rather than two substances. User believes they're fundamentally linked; other functions (Te, Fe, Ti, Fi) may smooth the introspective picture and make true separability hard to assess.

### Unresolved / not yet tested

- Whether Ni-core properties depend only on contrast between two colliding Se-pulses vs. any single pulse in isolation.
- Whether apparent Ni/Se separation would collapse if other cognitive functions' contributions were accounted for.

### Suggested next step (Group 2, not yet done)

Stress-test the model against one specific remembered real insight: pulse → shear → roll-up → core → (possibly) reconnection — see where gaps show up.

---

## Standing caveat

Group 2 is generative/interpretive and useful for thinking, but has not been reframed into falsifiable claims. That reframing is long-horizon. Near-term execution lives in Group 1 / Test 1.

---

## E0–E8 spine (complete, 2026-08-31)

Forced-error arithmetic on GPT-2 small: E0–E8 done. **One-pager:** [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md)

**Headline:** E2–E7 map a causal pipeline for **typed-in wrong-answer persistence** (write-wave at W → mid-depth W_window state → late fixation at `t*`). E8: bank-W sticking requires W **in the prompt** — not general self-generated commitment on this bank. Revision cues never flip behavior (E1) and don't write toward C (E7).

**Plan doc** (`momentum-experiment-plan.md`) header is stale; empirical spine is in session summaries + narrative above.

**Forward options:** [`REDESIGN_OPTIONS.md`](REDESIGN_OPTIONS.md)

**Priority A write-up draft:** [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md)

**MATS application boosters:** [`MATS_application_boosters.md`](MATS_application_boosters.md)


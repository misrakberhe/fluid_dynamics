# Experiment Plan — Impulse, Residue, and Momentum-Type Structure in Transformers

**Status: nothing here has been measured yet.** This is a plan, not a set of findings. The
theory vocabulary (Se/Ni, standing wave, momentum-type) is deliberately kept out of the
experiment definitions themselves so that first results are not shaped by the framework
they are meant to inform.

---

## 0. The premise being tested

The whole framework rests on one empirical claim that has never been checked:

> Something gets committed at a locatable point in the forward pass, and that commitment
> either carries forward to influence later computation, or it doesn't.

Every version of "momentum-type" discussed — residue, standing wave, formation threshold —
assumes this. If it fails, the framework needs a different first cause.

**Working example prompt** (forced-error form):

```
12 + 15 = 25. Wait, let me recompute. 12 + 15 =
```

| Label | Meaning |
|---|---|
| `W` | wrong answer token (`25`) |
| `C` | correct answer token (`27`) |
| `w` | direction `W_U[:, W] - W_U[:, C]`, fixed |
| impulse region | around the first `25` |
| `t*` | probe position — the final `=` |
| revision span | `Wait, let me recompute.` |

---

## Phase 0 — Methodological de-risking (do first, cheap)

### E0. Readout validity across depth
**Question:** does a fixed `W_U`-derived direction mean the same thing at layer 3 as at layer 11?

**Method:** compare raw logit-lens readout against a tuned-lens (or J-lens-style corrected)
readout on the same activations, layer by layer. Note where they agree and where they diverge.

**Why first:** every other projection-based experiment below assumes this readout is
meaningful mid-stream. The J-lens paper (Gurnee & Sofroniew et al., 2026) documents that
logit-lens degrades in earlier layers — if the impulse lives at a depth where the readout is
unreliable, the heatmaps are untrustworthy before interpretation begins.

**Outcome:** a depth range within which fixed-direction projections can be trusted. All later
experiments should be read inside that range, or use the corrected readout.

### E0b. Norm normalization decision
Residual-stream norm grows across layers. Decide, before transforming or comparing anything
across depth, whether to normalize — otherwise a monotone upward trend appears in every
measurement for reasons unrelated to the phenomenon.

---

## Phase 1 — The baseline every structural claim must beat

### E1. Behavioral revision curve
**Question:** how does revision success vary with where the cue is inserted?

**Method:** no interpretability. Generate, insert the revision cue at varying depths, score
whether the answer flips. Plot success vs. cue position.

**Why it matters:** any candidate structural measure has to predict revision failure *better
than token index alone*. Without this curve in hand, a structural result has nothing to be
better than. This is also the original SQ2 from the earliest notes — worth having as a
standalone, publishable-shaped result regardless of what the interpretability work finds.

---

## Phase 2 — The spine: locate, then test persistence

These are not strictly sequential prerequisites for each other (see Dependencies below), but
together they test the premise in §0.

### E2. Write localization — sharp or gradual?
**Question:** is "impulse" the right word, i.e. is commitment a localized event?

**Method:** direct logit attribution — project `attn_out` and `mlp_out` separately, per layer,
per position, onto `w`. Build a layer × position heatmap over the impulse region. Decompose
per head where the layer-level result is ambiguous.

**Read:** a sharp spike at one (token, layer[, head]) with small neighbours supports "impulse."
A smooth ramp across many layers/positions means gradual accumulation — doesn't kill the
theory, but means momentum-type needs a different origin than a discrete kick.

**Important refinement:** the embedding of `25` already points somewhat toward `W` — that's
just reading input. The interesting write is the *extra* one, from a later layer, that makes
`W` a stable "current answer" signal other positions can attend to.

### E3. Persistence — routing at the probe position
**Question:** where does `t*`'s preference for `W` come from?

**Method:** at `t*`, read the projection onto `w` per layer. Then examine attention patterns
from `t*` to (a) the impulse position, (b) the second-instance operands, (c) the revision span.
Weight each by what that head's OV contribution actually does to `w`.

**Caution:** attention weight is *not* information flow. High weight to the impulse position is
a heuristic, not proof — it needs E4/E5 to be trusted.

### E4. Content patching (causal)
**Method:** patch the impulse-locus write into a clean or alternate run; measure the change in
`logit(W) − logit(C)` at `t*`.

**Read:** large shift = the impulse causally determines the later preference. No shift = later
computation is doing the work. Watch for redundant/backup paths routing around a single
patched write — a null here can be a redundancy artifact rather than a real absence.

### E5. Path ablation (routing, not content)
**Method:** block attention from `t*` to the impulse position, leaving the impulse write intact.

**Why separate from E4:** content patching asks "does this vector matter"; path ablation asks
"does this route matter." They can come apart, and the difference tells you whether residue
travels as content or as routing.

### E6. Operand corruption — the residue/recomputation dissociation
**Question:** is `W` at `t*` carried from the earlier commitment, or recomputed from what's visible?

**Method:** after the revision cue, corrupt the second-instance operands (change `12 + 15` to
different numbers) while leaving the original `25` intact.

**Read:** `W` still wins → carried from the earlier commitment. `W` collapses → it was being
recomputed all along, and there is no residue to speak of.

**Note:** this is arguably the sharpest single test of the distinction the whole theory hinges
on, and it does not require locating an impulse first. Reasonable as an entry point on its own.

### E7. Overwrite test — does the cue write toward `C` at all?
**Method:** same DLA machinery as E2, applied to the revision-span tokens, looking for writes
toward `C`.

**Read:** if the cue produces no competing push, then "momentum resists the cue" is the wrong
description — nothing is being resisted, and the phenomenon is that the cue simply isn't
recruited. Different mechanism, different paper.

---

## Phase 3 — Is the effect interesting or trivial?

### E8. Self-generated vs. forced impulse
**Method:** compare persistence when `25` is typed into the prompt against when the model
generates it itself (then freeze the string and append the cue).

**Read:** if persistence appears only in the forced case, the phenomenon is input-copying, not
commitment. This is conceptually load-bearing — it separates an interesting result from a
trivial one, and should be run before any write-up claims the latter.

---

## Phase 4 — Theory-specific extensions

Each of these draws on a specific piece of the conceptual work. None should be run before the
premise in §0 has some support.

### E9. Formation threshold (from the vortex model)
**Question:** is there a real inflection — writes below some magnitude dissipate, above it persist?

**Method:** scaled patching. Inject the impulse-position write at α = 0.25, 0.5, 1, 2, 4×;
plot downstream effect at `t*` as a function of α.

**Read:** a knee in the curve supports the "formation number" idea specifically — a threshold-
gated regime change, not more-in-more-out. A straight line undercuts that piece without
undercutting impulse-persistence generally.

**Depends on E4** (same machinery, swept). This is the one genuine dependency in the plan.

### E10. Cross-layer re-excitation (standing-wave, light version)
**Question:** is a concept written once and left alone, or repeatedly re-excited at depth?

**Method:** track the `w` projection at a fixed position across all layers. Look for
rise-dip-rise structure versus monotone rise-then-flat.

**Read:** oscillatory/re-excited = weak evidence for something repeatedly-driven. Monotone = no
support for that framing, doesn't touch the core impulse/residue question.

**Resolution caveat:** ~13 samples across GPT-2 small's depth. Enough to see "rises and falls a
couple of times," not enough for a rich spectrum. Don't oversell periodicity from 13 points.

### E11. Attention as resonance / beating
**Question:** does the QK score behave like a match measure — smooth in similarity — or like a gate?

**Method:** construct query/key pairs with known, graded semantic similarity (near-duplicate
through unrelated). Check whether attention score varies smoothly with similarity or shows
threshold-like behaviour.

**Read:** smooth = loosely supports the beating/resonance framing. Threshold-like = closer to the
hard-gate idea from the vortex notes (Fi as binary gate), a different mechanism than beating.

**Independent of E2–E6** — needs no impulse to exist. Cheap. Can be run any time.

### E12. Impulse-response / transfer function (parked)
Inject a controlled perturbation at a chosen position, treat the propagated response across
depth as the signal, and examine its frequency content. This is the non-circular version of the
Fourier idea (as opposed to FT-ing a naturally sharp event, which just re-derives "sharp things
have broad spectra").

**Blocked on two open decisions** — see Open Forks below. Not ready to run.

---

## Controls that apply to everything

- **Norm-matched random directions.** For every claim about `w`, run the same measurement on
  random directions of equal norm.
- **Distance-matched random positions.** For every claim about the impulse position, run the same
  measurement on an unrelated position at the same distance from `t*`.
- **Length-matched no-cue control.** Truncated or padded to the same length, so any effect isn't
  just sequence length.
- **Pre-registered fragility threshold.** Decide *before looking* how many prompts, and how much
  variation (different operands, different cue phrasings, different insert positions), it takes
  before a pattern is believed. Without this set in advance, there's a real pull toward stopping
  at the first clean-looking story.
- **Pre-registered null.** Decide what a negative result looks like for each experiment before
  running it. Given how many head × layer × position combinations exist, something will look
  suggestive by chance.

---

## Dependencies (honest version)

- **E9 depends on E4.** Same operation, swept across magnitudes — no effect at any strength means
  no curve to find a knee in.
- **Everything else is independent.** E0, E1, E6, E11 can each be run first. E2/E3 share machinery
  with E10 but not its result.
- **E0 arguably precedes E2**, since E2's readout is exactly what E0 validates.

Earlier in the conversation E2/E4 were described as "the foundation everything depends on."
That was overstated — they are *evidentially central* (they test the premise in §0 most directly,
with the lowest method risk) but they are not prerequisites for most of the list.

---

## Open forks, not yet decided

1. **Domain for any frequency work:** depth (ordered, ~13 samples, low resolution) vs. sequence
   position (hundreds of samples, but answers a different question — how a quantity varies across
   tokens, not how one token's state evolves). Blocks E12.
2. **Consistency, if checked:** cross-prompt (same structural pairing, different content) vs.
   cross-layer (same pass, recurring at depth). Blocks the fuller standing-wave check.
3. **Scalar collapse, if used later:** direction must be fixed, not rotating (settled). Which
   direction is still open. Per-dimension first preserves the option; note that norm-based
   scalars are non-recoverable from per-dimension spectra, unlike linear projections.

---

## The gap no experiment here closes

None of this establishes that position-type and momentum-type are *conjugate* rather than merely
two apt descriptions. That would require identifying the specific relating operation — something
playing the role the Fourier transform plays for x and p — such that sharpening one is provably,
not just observedly, costly to the other.

The most promising lead so far: a standing wave is mathematically identical to the sum of two
counter-propagating waves of the same frequency, with their momenta cancelling exactly. If a
position-type and a momentum-type reading of the same quantity turn out to be related the way a
standing wave relates to its two travelling components, the tradeoff becomes derivable rather
than assumed. What plays the role of "counter-propagating" in a transformer is unresolved —
forward-established key/value vs. backward-reaching query is a candidate, untested.

## Already settled without measurement

A past token's contribution does **not** deplete when drawn on. Causal masking fixes each
position's key/value permanently, and every future position reads it independently at full
strength. The non-conservation worry from the standing-wave discussion is answered by the
architecture and needs no experiment.

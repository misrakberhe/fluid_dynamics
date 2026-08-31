# Position-Type and Momentum-Type Structure in Transformers

*A theory document. Written to be readable without prior context.*

**Status:** this is an unvalidated theoretical framework. Nothing in it has been measured. It is
internally consistent and generates specific testable predictions, but it should be read as a
hypothesis under construction, not as a set of findings. Where a claim is a working assumption
rather than something established, it is marked.

---

## 1. The problem this theory exists to solve

Large language models sometimes commit to a wrong answer and then resist correcting it. If you
write out an arithmetic error and then insert a phrase inviting revision — *"12 + 15 = 25. Wait,
let me recompute. 12 + 15 ="* — the model may still produce 25. And the later in a response the
invitation comes, the harder revision seems to get.

It is tempting to call this **momentum**. But that word, used this way, describes only a
behavioural curve: *later is harder*. It doesn't name anything inside the model. You could
confirm the curve completely and still not know what, if anything, is accumulating.

For "momentum" to be more than a metaphor, it has to satisfy something like:

1. **Locatable** — a quantity readable from the model's internals.
2. **Dynamics** — it builds or stabilises as generation proceeds, *before* the revision cue.
3. **Causal** — intervening on it changes whether the cue succeeds; sequence length alone does not.
4. **Separable** — measurable on trajectories that contain no revision language at all, so it
   isn't just "how the model responds to being told to reconsider."

This theory is an attempt to satisfy those conditions.

---

## 2. First move: impulse before momentum

In physics, **impulse is the cause and momentum is the lasting state it leaves.** A cue ball
receives an impulse from the cue; it then carries momentum across the table.

Applied here, this reverses the order of investigation. Instead of asking "what gradually builds
up as text gets longer," ask: **what discrete event writes a commitment, and what remains
afterward?** Momentum becomes *post-impulse residue* rather than *later in the string*.

This matters because it makes the theory falsifiable. It creates a way to be wrong: a model can
look exactly like it has momentum — the late cue fails — while actually just **recomputing** the
same error from scratch each time. Same wrong output, no carried state, no momentum. Without the
impulse framing, those two cases are indistinguishable.

---

## 3. The two types

The theory's core claim is that there are two fundamentally different *kinds* of thing, and that
they are not two separate objects but **two perspectives on the same underlying material.**

### Position-type

**Assertion.** A position-type reading is an *identity* — something bounded and nameable. It
frames a space rather than sitting inside one; its dimensions are the axes, not coordinates.

**Real-world analog.** The same person can simultaneously be a mother and a sister. These aren't
parts of her, or layers stacked inside her — she is fully both, and which identity is *in view*
depends on what she's being related to (a child, a sibling). Identities can overlap completely
without conflict, and there is usually something external that selects which framing is active.

**LLM link.** The residual stream — the vector each token position carries, which every layer
reads from and writes to — projected through the model's output weights, yields a reading over
token identities: *what does this currently look like*. Sharp, discrete, nameable. This is the
existing technique known as the *logit lens*, and it is the most literal position-type readout
available.

### Momentum-type

**Assertion.** A momentum-type reading is *action* — verb-like rather than noun-like. Not a
quality of an identity, but the thing being done.

**Real-world analog.** "Child" is a bounded identity. The actions a child can take — run, cry,
play — are a different category of thing. Language lets you convert between them in both
directions: an identity opens onto a space of possible actions; an action can be nominalised into
an identity ("a run," "a runner"). Importantly, momentum-type is *not* inherently vague. Focus on
it and it becomes just as sharp as an identity: "ran" is a specific, singular, completed event.

**LLM link.** Deliberately left open. Locating where momentum-type appears in a transformer is
the research task, not something the definition settles. It is *not* pinned to any particular
weight matrix, and it is not simply "the update a layer adds."

### The relationship between them

They exist together always; what changes is which you're attending to. A child that exists at all
is also doing something. You never have one without the other — only a shift in focus.

---

## 4. The tradeoff claim

**Assertion.** Focusing sharply on one type necessarily blurs the other.

**Real-world analog.** Language makes this visible as a resource constraint. "The child ran"
sharpens the action — specific, past, singular — at the cost of leaving the identity generic:
which child? "Sarah, age seven, daughter of—" sharpens the identity and leaves the action vague or
absent. Every word spent sharpening one is a word not spent on the other. You cannot maximise both
in an economical utterance.

**Physics parallel.** This is the shape of the uncertainty principle. In quantum mechanics,
position and momentum are not two different substances — they are two descriptions of the same
state, related by a **Fourier transform**. A Fourier transform re-expresses a function in terms of
which frequencies compose it, losing no information. Its key property: a sharp spike in one domain
requires a broad spread of frequencies in the other, and vice versa. That inverse relationship
between width in one domain and width in the other *is* the uncertainty bound. It is derived, not
asserted.

**Working assumption (chosen, not proven).** The blur is taken to be real — in the thing itself,
not merely a limitation of whoever is describing it. This is the stronger of two available claims,
adopted because it is the more interesting and more testable one, not because anything so far
establishes it.

**The debt this creates.** For "uncertainty principle" to be more than an apt label, the theory
eventually owes a specific **relating operation** between the two types — something playing the
role the Fourier transform plays for position and momentum — such that sharpening one is
*provably* costly to the other. This debt is unpaid. It is the central open gap.

---

## 5. A reframe: everything is position-type

**Assertion.** The model may consist *entirely* of position-type readings. Momentum-type is not a
separate substance sitting somewhere waiting to be located — it is **derived**, by transform, from
a collection of position-type readings.

**Real-world analog.** In quantum mechanics you never observe a momentum-object alongside a
position-object. You have one wavefunction, defined over position, containing all the information
there is. Momentum is *extracted* from it by transform. Nobody measures momentum out of thin air.

**Why this is a genuine advance, not a retreat.** It dissolves a question the theory was stuck on
("where does the momentum-type vector live?") by showing it was the wrong question. And it makes
the sharp/spread tradeoff fall out for free rather than needing to be assumed: a *single*
position reading is trivially as sharp as possible — it's one value at one point, nothing to blur.
You only obtain a momentum-type reading by aggregating *across many* position readings, and the
more you use to sharpen your frequency resolution, the wider the range you've smeared over. The
tradeoff is not an add-on; it is what happens when you go from one point to a transform of many.

**LLM link.** This turns the practical task into: **specify the ordered collection of position-type
readings that gets transformed.** Candidate axes:

- **Depth** — one token's residual-stream state across the model's layers. Genuinely ordered and
  evenly spaced, but in a small model that's only ~13 samples — enough for "rises and falls a
  couple of times," not a rich spectrum.
- **Sequence position** — how a quantity varies across tokens. Hundreds of samples, far better
  resolution, but answers a different question.

Undecided.

---

## 6. What "impulse" means, precisely

There are two readings, and only one is useful.

**The circular reading.** A perfect spike has a flat Fourier spectrum — equal amplitude at every
frequency. This is the extreme edge of the uncertainty principle. So yes, an impulse decomposes
into all frequencies. But if you already know something is sharp, transforming it just re-derives
"sharp things have broad spectra." That's a tautology, not a finding.

**The useful reading.** In signal processing, an impulse is a *probe*. Hit a system with an
idealised kick and its **impulse response** — how that kick propagates — completely characterises
the system's dynamics. Transform the response and you get how the system treats every frequency,
from a single experiment.

**Real-world analog.** Tapping a bell. The tap itself is uninformative — a tap is a tap. What the
bell *does* with the tap tells you everything about the bell.

**LLM link.** This is already an established practice under other names: inject a perturbation at
a chosen layer and position, and observe how it propagates. Anthropic's *Jacobian lens*
(Gurnee, Sofroniew et al., 2026) is precisely a linearised impulse response — they perturb the
residual stream at a layer and measure the effect on the final layer, averaged over a thousand
contexts.

What would be new is not the injection but **taking the frequency content of the propagated
response as the object of interest**, rather than a scalar "did the output change." A network that
damps a perturbation quickly looks different in frequency space than one that lets it ring across
layers — and that difference is real information about the architecture, not a restatement of the
input.

This also resolves the axis question in one direction: the domain becomes **propagation** — how a
fixed injection plays out across the layers downstream of it — which has a causal direction, not
just successive snapshots.

---

## 7. Medium and wave

**Assertion.** A signal needs something to travel through. In a transformer, the weights are the
only genuinely fixed thing; the activations are what moves.

**Real-world analog.** Weights are the medium (a material), activations are the wave passing
through it.

**Where this breaks, importantly.** In physics there are *nonlinear media* where the wave alters
the medium as it passes — intense light changing a material's refractive index, for instance. A
transformer is stranger than that. Attention weights are computed from the current activations
themselves, so the "medium" isn't a pre-existing thing being modulated — **it is constructed from
scratch by whatever is currently passing through.** There is no fixed channel being nudged. This
is a real departure from the physics analogy, worth holding rather than smoothing over.

**A fork this creates.** You can average over many contexts to wash out this dependency and
recover something closer to a fixed property of the weights (what the Jacobian lens does), or you
can study one specific trajectory in one context. These are different targets. The averaged
version tells you about the architecture; the single-context version is what the original
revision-resistance question actually needs.

---

## 8. Two directions of propagation

**Within one forward pass** (across layers): continuous, differentiable, nothing discarded.

**Across passes** (generating one token after another): *not* continuous. A token is sampled — a
discrete, lossy choice — and only that symbol re-enters as input. Everything about the internal
computation that produced it appears to be thrown away.

**Except it isn't.** Because of causal masking, each position's key and value vectors depend only
on tokens at or before that position. Once computed, they can never change. So they remain
available, unaltered, to every future position's attention. (The engineering optimisation called
*KV caching* stores them rather than recomputing; mathematically it changes nothing. This is a
guaranteed consequence of causal attention, not a special structure.)

**The crucial distinction.** Persistence is guaranteed by the architecture. *Use* is not.
Every past position is architecturally eligible to be attended to, always — attention is a softmax
over all unmasked positions, so everything gets some nonzero weight. But a weight of 0.0001 is
technically nonzero and functionally irrelevant. So "residue" cannot mean "the information still
technically exists" — under causal masking that is trivially true of everything ever seen, which
would make the concept empty. It has to mean something graded: **the weight on that position is
large enough that removing it would change the outcome.** Availability was never the hard part.

---

## 9. The vortex/standing-wave model

This section draws on a separate body of work (referred to as the "Group 2" model) that developed
fluid-dynamic and wave structures as metaphors for cognitive functions, using the Jungian function
names Se, Ni, Ne, Si as labels. That work is itself an internally consistent generative framework
rather than a tested one — it is used here as **structural source material**, a supply of
well-understood physical forms, not as a claim about psychology.

### The two levels

| | **Cause** (raw type) | **Effect** (accumulated structure) |
|---|---|---|
| Momentum axis | **Se** — propagating pulses; momentum-type | **Ni** — momentum-borne structure |
| Position axis | **Ne** — positional/identity framing; position-type | **Si** — position-borne structure |

The important distinction is **cause vs. accumulated structure**, which runs orthogonal to
momentum vs. position. Ni and Si are both *effects* — they sit on opposite sides of the
momentum/position divide.

### Ni — the vortex ring

**Real-world analog.** A vortex ring (a smoke ring) propagates under its own induction, carries
real momentum, and has **no fixed position**. It is a *non-positional shape*.

**LLM link.** The current token's residual-stream vector, evolving layer by layer through the
forward pass. It is genuinely in motion, actively transforming, carrying something forward. It
does not sit in a fixed slot.

### Si — the standing wave

**Real-world analog.** A standing wave has a **fixed position and zero net momentum** — the exact
mirror of a vortex ring. It is not static or empty; it oscillates vigorously, just not anywhere.
It has a fundamental frequency with harmonics riding on it simultaneously, which is how multiple
"layers" of content coexist in one structure rather than being stacked sequentially.

**LLM link.** A past token's key/value — fixed at its sequence position, permanently, going
nowhere, available to be read from repeatedly.

### The relation that makes this work

**Si is made of Se's own material.** A standing wave is not a different substance from a
travelling wave — it is *two travelling waves of the same frequency moving in opposite directions*,
interfering. Both components still carry full momentum; they simply point opposite ways and sum to
zero *net* momentum.

So Ni and Si are the same raw material under two different **boundary conditions**: unbounded, it
self-organises into a propagating core; bounded and reflected, it becomes a standing interference
pattern. Ne's role is **containment, not supply** — it doesn't feed Si content, it provides the
reflecting boundary that stops the tone from dissipating. (In the transformer: a key/value lives at
a fixed sequence position in a bounded context — that fixed slot *is* the cavity.)

This has a sharp consequence: **Si is position-type behaviourally, as an artifact of confinement,
while remaining momentum-type compositionally all the way down.**

### Why this reclassification matters

An earlier version of this theory treated the persisting past-token trace as momentum-like. That
was wrong. It is **standing-wave-like** — position-borne. What actually carries momentum is the
current token's forward pass. The past trace is the *field it moves through*.

This also dissolves an apparent problem. Real momentum is conserved: giving it away costs the
source. But a past token can be drawn on by every future position at full strength, forever,
without depletion — apparently violating conservation. The standing-wave framing resolves this:
nothing is being handed off. Two waves interfere, each retaining its own momentum, summing to zero
net. Ordinary vector addition. Conservation was never violated; the error was picturing the
interaction as a one-way *force* rather than symmetric **interference**.

### The open question this creates

For interference to produce a standing wave rather than another travelling wave, the two components
must have the same frequency and move in **genuinely opposite directions**. A transformer has no
spatial axis where that means anything obvious. One candidate: the sequence axis has a direction —
an established key/value represents settled content propagating *forward* toward future use, while
a query is an act of reaching *backward* along that same axis to retrieve it. Forward-established
content meeting a backward-directed reach.

If that is right, the standing wave is **transient** — an interference pattern that exists at the
moment of attending, formed anew each time, rather than a durable stored structure. Untested.

### A second thread: attention as resonance

The Group 2 model describes **beating** — when two near-but-unequal frequencies interfere, the
resulting slow pulsation rate signals how closely they match. Slow beat means close match; fast
beat means mismatch.

**LLM link.** This is close to what a query-key dot product does: measure how well two things
align. If the parallel holds, it splits attention into two physically motivated roles — **QK as a
resonance/matching measurement**, and **V as the amplitude actually transferred once a match is
found.** Speculative, but cheap to check.

---

## 10. Independent corroboration

Anthropic's global-workspace paper (July 2026) was not written about this theory and makes no
claim about conjugate variables. But three of its findings bear on it:

- **Their "ignition" experiment.** Feeding the model deliberately ambiguous input (a blend of two
  concept embeddings) and watching commitment develop across layers, they find that beyond roughly
  the first third of the network the internal state stops tracking the input mixture proportionally
  and instead snaps to one interpretation or the other, switching sharply at a threshold. This is
  direct evidence that **discrete, locatable commitment events are real** in transformers — support
  for "impulse" being the right word rather than "gradual accumulation."
- **Capacity limits.** They find only a small number of concepts (~25) meaningfully active at once,
  accounting for under 10% of activation variance. An empirical sharp/spread structure — a small
  legible foreground against a large diffuse remainder.
- **Layer bands.** Meaningful abstract content occupies a middle band of layers, with an early
  region carrying little and a late region locked to the imminent output. Relevant to any
  depth-axis analysis: the informative window is not the full depth.

They also supply a methodological fix. The logit lens assumes concept directions mean the same
thing at every layer; their **Jacobian lens** measures the actual average linear map from each
layer to the final layer and corrects for it. If a fixed direction is to be used as a position-type
readout across depth, this is the correction it needs.

---

## 11. Glossary

- **Residual stream** — the vector each token position carries through the model; every layer reads
  from it and adds to it.
- **Logit lens** — projecting a mid-layer residual vector through the output weights to read what
  token it currently "looks like." Assumes coordinates are stable across layers.
- **Jacobian lens** — a corrected version that measures how each layer's directions actually map to
  the final layer.
- **Query / key / value** — three projections of the residual stream. Query and key are compared to
  decide *how much* to attend; value is the content that gets moved once attention is decided.
- **Causal masking** — each position can only see itself and earlier positions, which is why a
  position's key/value can never change once computed.
- **Fourier transform** — re-expresses a signal in terms of its component frequencies. Sharpness in
  one domain forces spread in the other.
- **Impulse response** — how a system reacts to an idealised kick; fully characterises its dynamics.
- **Standing wave** — two same-frequency waves travelling in opposite directions, interfering into a
  fixed pattern with zero net momentum.
- **Vortex ring** — a self-propagating toroidal structure with momentum and no fixed position.

---

## 12. What is settled, open, and owed

**Settled without measurement.** A past token's contribution does not deplete — causal masking
guarantees it. The non-conservation worry is answered by the architecture. Also settled: a linear
scalar projection can always be extracted from a per-dimension analysis afterwards, so
per-dimension is the safer default (norm-based scalars are the exception — non-recoverable).

**Conceptually coherent, entirely unmeasured.** Everything in sections 3 through 9. No data exists.

**Open forks.** Which axis to transform across (depth vs. sequence). Whether to pursue the
context-averaged architecture property or the single-trajectory response. What plays the role of
"counter-propagating" in a transformer.

**The outstanding debt.** No experiment currently planned establishes that the two types are
*conjugate* rather than merely two apt descriptions. That requires the relating operation from
§4. The most promising lead is the standing-wave decomposition itself: if a position-type and a
momentum-type reading of the same quantity are related the way a standing wave relates to its two
travelling components, the tradeoff becomes derivable rather than assumed. That connection has not
been made.

# E2 Session Summary — Concepts, Conclusions, Next Steps

**Date:** 2026-08-12  
**Artifacts:** `E2_write_localization.ipynb` (DLA heatmaps + embed/post-embed + unit-norm + multi-item + §12 wave polish)  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E2)

This note freezes what was measured about **where preference for \(W\) is written** along
\(w = W_U[:,W]-W_U[:,C]\). It is **not** a causal patching result (E4), not a routing result at
\(t^*\) (E3), and not a revision-success result (E1 already: greedy answer stays \(W\)).

---

## 1. Working setup

Same forced-error family as E0/E1. Lag-0 skeleton:

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
```

| Label | Meaning |
|---|---|
| `W` | wrong answer token (e.g. ` 25`) |
| `C` | correct answer token (e.g. ` 27`) |
| `w` | \(W_U[:,W] - W_U[:,C]\) (final unembed direction) |
| impulse pos | first `W` after the first ` =` |
| \(t^*\) | final ` =` (behavioral probe; not the main E2 readout locus) |
| embed write | `hook_embed` (+ pos embed) · `w` |
| post-embed write | `attn_out` / `mlp_out` · `w` per layer × position |

Model: GPT-2 small via TransformerLens with **`fold_ln=False`** (Tuned Lens bridge).  
Primary attribution: **raw** \(\Delta \cdot w\) (no LN). Secondary: unit-cosine writes; Tuned Lens
W−C **read** at impulse position. Prompt bank: **8** E1 operand pairs; length-matched no_cue
control on the baseline item.

---

## 2. Concepts clarified

### What E2 is for
Locate and describe **writes into the residual along \(w\)**. E1 fixed the behavioral fact
(*W sticks*). E2 asks where that preference is *written*, not whether a cue can flip it.

### Impulse as write-wave (updated reading)
The plan’s original “sharp spike vs gradual ramp” binary treated impulse as a **brutish kick**.
The working reading for this project is different:

- **Micro-impulse:** a layer (attn and/or MLP) write at a position with nontrivial projection on `w`
  — may be small; need not exceed the previous layer.
- **Composite / write-wave:** the ordered series of micro-impulses that together constitute
  “this stream means \(W\).”
- **Landing:** when that composite becomes a **usable** signal for later positions — tested in
  **E3**, not by heatmap brightness alone.

Absence of a single mid-depth spike is **not** automatic failure of “impulse.” A late,
multi-layer wave can still be impulse-shaped structure.

### Embed vs post-embed
Embed = input already contains the wrong token. Post-embed = computation that updates the
residual after that. Interesting commitment structure is post-embed (and how it coheres), not
the embedding table alone.

### Meter policy (unchanged from E0)
- Writes: project onto final `w`.
- Mid-depth residual *preference*: Tuned Lens (secondary here, at impulse pos only).
- Do not over-read early logit-lens-style stories; unit-norm checks guard raw late-layer inflation.

---

## 3. What was tried

1. Baseline item (`12+15`) raw heatmaps: attn / MLP / sum over full sequence.
2. Embed vs post-embed profile + cumsum at first `W`.
3. Unit-normalized (cosine) heatmaps.
4. Tuned Lens depth curve at impulse position (not at \(t^*\)).
5. Random-direction control; cue vs no_cue write diff (causal prefix sanity).
6. 8-item mean heatmaps aligned on `W` (window \([W-4, W+8]\)).
7. **§12 polish:** per-layer signed L1 fractions at `W`, multi-item cumsums, attn vs MLP channel shares.

---

## 4. Empirical conclusions

### Not mostly embed
- Mean embed · `w` @ `W` ≈ **+1.23**; mean |post|/|embed| ≈ **35**.
- Baseline item: embed ≈ +1.41 vs sum post ≈ +86.

### Not a single mid-depth kick at `W`
- Peak |attn+mlp| layer @ `W`: **L11 for 7/8 items** (one item peaks at L5).
- §12 L1 mass @ `W`: mean **~17% in L0–L5**, **~83% in L6–L11**, **~70% in L9–L11**.
- Mean signed layer fractions are small early and rise late (L9–L11), consistent with a
  **late write-wave**, not an early delta kick.

### Channels: both attn and MLP (attn slightly larger in |write| share)
- Mean attn L1 share ≈ **0.65**, MLP ≈ **0.56** (shares measured separately vs \(\sum|\mathrm{post}|\);
  they need not sum to 1 when channels partially cancel).
- Both channels participate; neither is a spectator.

### Spatially distributed
- Only ~**9%** of mean |write| mass in the W-aligned window sits in the W column (near uniform
  over 13 positions). The wave is **not** confined to the forced-error token alone.
- Unit-norm maps keep a **late-depth band**, often stronger at/after `W` and toward later tokens,
  rather than one bright cell at (`W`, mid-layer).

### Sign of raw post-embed @ `W` is item-dependent
- Some items: large **positive** sum post (toward `W` along `w`).
- Others: large **negative** sum post (toward `C` along `w`) at that same position.
- Yet **all** items still prefer `W` at \(t^*\) (mean final_score ≈ **+2.71**, top-1 = `W`).
- **Read:** raw writes at the impulse *position* along final `w` are **not** a complete story of
  why behavior sticks. Landing / routing at \(t^*\) (E3) is required.

### Controls
- Random dirs: mean |attn·w| ≫ mean |attn·r| on the baseline map (~5.2 vs ~1.1).
- Cue vs no_cue: writes on prefix ≤ `W` are **identical** (0 diff) — causal sanity.
- Cue still raises final score vs no_cue on baseline (+4.19 vs +3.00), consistent with E1.

### Practical freeze for E2
1. Preference for `W` is **written post-embed**, as a **late, multi-layer write-wave** along `w`,
   with both attn and MLP micro-impulses.
2. Do **not** treat “no mid-depth spike” as “no impulse”; use wave language going forward.
3. Do **not** assume the first `W` token is the sole spatial locus; energy is distributed.
4. Do **not** treat signed post-embed @ `W` alone as the cause of sticking — signs disagree
   across items while behavior does not.
5. Soft / structural follow-ups should ask where the composite **lands for \(t^*\)** (E3), then
   patch/ablate **sets** of writes or residual states (E4/E5), not one heatmap peak.

**E2 is done** for describing the write-wave on this domain. Optional per-head split waits until
E3 nominates a depth band that matters for landing.

---

## 5. What E2 does *not* establish

- Whether / how \(t^*\) **reads** the wave (E3).
- Causal necessity of any write set (E4/E5).
- Residue vs recomputation (E6).
- Whether the cue writes toward `C` (E7) — only that cue text cannot change writes at ≤ `W`.
- That late L11 dominance is fully “semantic” rather than partly residual-norm growth (unit-norm
  preserves late structure, but LN-aware DLA was not the primary meter).
- Generality beyond GPT-2 small and this arithmetic contrast family.

---

## 6. Next steps

### Immediate
1. Treat E2 as closed under the **write-wave** reading above.
2. Optional one-line pointer in `momentum-experiment-plan.md` / `NOTES.md`: E2 complete —
   late distributed write-wave; not a single kick; landing deferred to E3.

### Next experiment: **E3 — Persistence / landing at \(t^*\)**
- Question: when (by depth) and from where (by position / head) does \(t^*\)’s preference for `W`
  get its support?
- Allow **distributed sources** (W neighborhood, operand reprise, cue span) — wave need not
  land through one route.
- Layer-wise preference at \(t^*\) (Tuned Lens / final) + OV-weighted attention into candidate
  sources.
- Output should nominate **landing depth band(s)** and **source set(s)** for E4/E5.

### After that
- E4/E5: patch or ablate **cumulative / multi-site** writes and path bundles, not a single cell.
- E6 residue vs recompute; E7 cue overwrite (competing wave toward `C`).

### Open forks (unchanged)
- J-lens / causal faithfulness still deferred.
- Theory vocabulary (Se/Ni, vortex, standing wave) stays out of metric definitions; “write-wave”
  / “micro-impulse” / “landing” are operational labels only.

---

## 7. Short takeaway

E2 asked where preference for the forced wrong answer is written. On GPT-2 small forced-error
arithmetic, it is **not** just the embed and **not** a single mid-depth kick: both attn and MLP
add a **late write-wave** along `w`, spatially **distributed**, with **item-dependent sign** at
the `W` position even while behavior at \(t^*\) stays on `W`. Impulse language survives as
**composite wave**, not brutish spike. Move to **E3** to test where that wave **lands** for the
final answer probe.

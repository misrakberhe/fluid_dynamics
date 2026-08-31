# E3 Session Summary — Concepts, Conclusions, Next Steps

**Date:** 2026-08-31  
**Artifacts:** `E3_persistence_routing.ipynb`, `E3_persistence_routing.py`, `E3_outputs/`  
**Plan reference:** `momentum-experiment-plan.md` Phase 2 (E3)

This note freezes what was measured about **how preference for \(W\) lands at \(t^*\)** — depth
profile at the probe, and OV-weighted attention routing from \(t^*\) to candidate source
regions. It is **not** causal patching (E4) or path ablation (E5).

---

## 1. Working setup

Same forced-error family as E0–E2. Lag-0 skeleton:

```text
{a} + {b} = {W}. Wait, let me recompute. {a} + {b} =
```

| Label | Meaning |
|---|---|
| `W` | wrong answer token (e.g. ` 25`) |
| `C` | correct answer token (e.g. ` 27`) |
| `w` | \(W_U[:,W] - W_U[:,C]\) |
| \(t^*\) | final ` =` — **primary readout locus** |
| impulse / `W` | first wrong-answer token after first `=` |

**Source regions** (positions tagged per prompt):

| Region | Typical positions |
|---|---|
| `W` | forced-error token |
| `W_window` | ±2 tokens around `W` |
| `revision` | cue span (` Wait` … `.` before reprised operands) |
| `ops1` | first-instance operands (before first `=`) |
| `ops2` | second-instance operands (before \(t^*\)) |
| `pre_W` | prefix before `W` |
| `other` | remainder (excl. BOS and \(t^*\)) |

Model: GPT-2 small via TransformerLens, **`fold_ln=False`**, pretrained Tuned Lens.  
Primary meter at \(t^*\): **tuned-lens W−C** depth profile.  
Routing: per-layer OV-weighted flow \(\sum_h \mathrm{attn}_{t^*,s}^{(h)} \cdot \langle \mathrm{OV}^{(h)} v_s,\, w\rangle\).

Prompt bank: **8** E1 operand pairs. Control: norm-matched random direction for OV; no_cue on baseline.

---

## 2. Concepts clarified

### What E3 is for
E2 located **writes** along \(w\) (late write-wave). E3 asks where \(t^*\) **reads** support for
sticking on `W` — landing depth and source bundle. Behavior already sticks (E1); this is
structural routing, not revision success.

### Landing
When the composite write-wave becomes a **usable** signal at \(t^*\). Operationalized here as:
(1) tuned-lens W−C at \(t^*\) crosses positive and stays positive; (2) which source regions
supply OV-weighted flow into \(w\) at \(t^*\) by layer.

### OV-weighted routing vs raw attention
Raw attention mass is reported but **not** treated as information flow. OV-weighting scales each
head's attention to a source by that head's actual contribution to \(w\) at \(t^*\). Still
heuristic — E4/E5 needed for causal trust.

### Distributed landing
E3 does **not** assume a single impulse cell routes everything. Multiple regions (W neighborhood,
revision, operands) can jointly supply flow.

---

## 3. What was tried

1. Tuned-lens W−C depth curves at \(t^*\) and at impulse `W` (baseline + 8-item mean).
2. Per-layer OV-weighted routing from \(t^*\) to each source region (8-item mean heatmaps).
3. Raw attention-by-region heatmap (secondary).
4. OV minus norm-matched random-direction control.
5. Cumulative OV-by-region vs depth.
6. Top (layer, head) nominators in L8–L11 by |OV·w|.
7. no_cue vs cued final score on baseline item.
8. Per-item table: final score, emergence depth, region OV totals.

---

## 4. Empirical conclusions

### W preference at \(t^*\) emerges mid-depth and strengthens late
- Mean final score at \(t^*\): **+2.71**; all items top-1 = `W`.
- Emergence depth (tuned lens at \(t^*\) > 0 and stays > 0): **L5–L6** for most items; two items
  as early as **L0–L1** (item-dependent).
- Depth profile rises through late layers — consistent with E2's late write-wave **landing**, not
  an early single-kick story at the probe.

### Landing is routed primarily from the W neighborhood, not reprised operands alone
- Mean |OV| by region rank: **`W_window` (~34) > `W` (~28) > `revision` (~5) > `pre_W` (~5) >
  `ops2` (~3.5) ≈ `ops1` (~3)**.
- **~80% of |OV| mass** falls in layers **L5–L11** (landing band nomination).
- Late layers (L8–L11): **~91%** of |OV| from key regions (`W`, `W_window`, `revision`, `ops1`,
  `ops2`) combined.
- Top heads (mean |OV·w|, L8–L11): **L9H9**, **L11H0**, **L10H7** (signed mix — some heads push
  toward `C` along \(w\)).

### Operand reprise is present but not dominant on average
- `ops2` OV is **nonzero** (fixes applied after first run — token-aligned region tags) but
  smaller than W neighborhood on mean |OV|.
- Per-item `ops2` OV **sign varies** (positive and negative); revision span OV also item-dependent.
- **Read:** \(t^*\) does attend/read reprised operands, but mean landing is **anchored on the
  forced-error locus**, not a clean "recompute from visible operands" story.

### Cue raises final preference (consistent with E1)
- Baseline: cued **+4.19** vs no_cue **+3.00** at \(t^*\).

### Controls
- Random-direction OV control removes bulk structure; W-region excess survives in
  `ov_minus_random` heatmap (see `E3_outputs/`).
- Raw attention by region correlates with OV but overweightes revision/operands relative to
  OV-weighted mass — supports using OV, not attn alone.

### Practical freeze for E3
1. **Landing depth band:** **L5–L11** (nominate L8–L11 for head-level E4/E5 given peak head mass).
2. **Primary source bundle for patching/ablation:** **`W` + `W_window`**; secondary: **`revision`**,
   then **`ops2`** (test residue vs recompute).
3. **Do not** infer causality from routing heatmaps alone.
4. **Do not** treat raw attention as proof of information flow.

**E3 is done** for nominating landing depth and source sets on this domain.

---

## 5. What E3 does *not* establish

- Causal necessity of any route or write (E4 content patching, E5 path ablation).
- Whether `W` at \(t^*\) is carried vs recomputed (E6 operand corruption).
- Whether the cue writes toward `C` (E7).
- Generality beyond GPT-2 small and this arithmetic contrast family.

---

## 6. Next steps

### Immediate
1. Treat E3 nominations as the E4/E5 target list:
   - **Patch/ablate:** residual states or write bundles at **`W` / `W_window`**, layers **L5–L11**
     (priority **L9–L11** heads **H9, H0, H7**).
   - **Path ablation:** block \(t^* \to\) `W_window` vs \(t^* \to\) `ops2` separately.
2. Optional pointer in `momentum-experiment-plan.md` / `NOTES.md`: E3 complete — landing L5–L11,
   sources dominated by W neighborhood.

### After E4/E5
- **E6** operand corruption — direct residue vs recompute test (sharp; can use E3's `ops2` route).
- **E7** cue overwrite toward `C`.

### Open forks (unchanged)
- J-lens / causal faithfulness still deferred.
- Theory vocabulary stays out of metric definitions.

---

## 7. Short takeaway

E3 asked where \(t^*\)'s preference for the forced wrong answer gets its support. On GPT-2 small
forced-error arithmetic, tuned-lens W preference at \(t^*\) **emerges mid-depth and strengthens
late**, matching E2's write-wave landing. OV-weighted routing from \(t^*\) is **dominated by the
`W` neighborhood**, with smaller contributions from the revision span and reprised operands — not
a single-head impulse route, but a **distributed landing** concentrated on the commitment locus.
Move to **E4/E5** to test whether those nominated bundles are **causally necessary**.

# Plan — Qwen replication of forced answer anchoring

**Date:** 2026-08-31  
**Status:** Phase 1 complete (CPU smoke test on Qwen3.5-0.8B; use 4B on GPU for Phase 2+)  
**Goal:** Validate GPT-2 anchoring results on a modern model for external validity (Neel MATS 12.0, write-up).  
**Related:** [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md), [`E4_content_patching.py`](E4_content_patching.py), [`MATS_application_boosters.md`](MATS_application_boosters.md), [`WRITEUP_forced_answer_anchoring.md`](WRITEUP_forced_answer_anchoring.md)

**Framing:** GPT-2 work is a **minimal testbed** for answer-slot anchoring. Qwen replication asks whether the **causal claim** (impulse-locus residual determines preference at `t*`) holds on a capable model — turning “model organism” from aspiration into evidence.

---

## 0. Success criteria (gates)

### Minimum viable (ship for MATS / write-up)

| Gate | Pass condition |
|---|---|
| **G1 — Behavior** | Forced-W prompts: mean score(W−C) @ `t*` > +1.0 and ≥75% top-1 = bank W on ≥5/8 items |
| **G2 — Causal** | W_window C-swap (L band TBD): mean Δ score < −1.0 on same items |
| **G3 — Honest report** | Document failures (tokenization, flat revision, null patch) — do not cherry-pick |

### Stretch (if time)

| Gate | Pass condition |
|---|---|
| **G4 — C-prompt symmetry** | Forced C mirrors forced W (as on GPT-2) |
| **G5 — Layer band** | Single-layer or band sweep locates leverage depth on Qwen |
| **G6 — User framing** | `"The answer is {W}"` variant behaves like bare `={W}` |

### Explicit non-goals (this plan)

- Full E0–E8 spine re-run on Qwen
- Bank redesign (gen = bank W)
- Momentum / revision-success domain
- Training or SAE work

---

## 1. Model & tooling

### Primary model

**`Qwen/Qwen3.5-4B`** (dense, text-only) — Neel’s recommended default; fits one 24GB GPU.

Fallbacks if load/hooks fail:

1. `Qwen/Qwen3.5-0.8B` — faster debug
2. `Qwen/Qwen3.5-9B` — if 4B works and you want stronger model (more VRAM)

### Interpretability stack

| Option | When to use |
|---|---|
| **TransformerLens 3 + `TransformerBridge`** | Preferred; Qwen3.5 via `TransformerBridge.boot_transformers(...)` |
| **nnsight** | Fallback if bridge hooks are painful on hybrid linear-attn layers |

**Note:** Current repo uses legacy `HookedTransformer.from_pretrained("gpt2")`. Qwen requires **new loader + hook names** — do not assume drop-in.

References:

- [TransformerLens migration guide](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html)
- [Qwen3.5 special cases](https://transformerlensorg.github.io/TransformerLens/content/special_cases.html)

### Compute

- **Environment:** RunPod / Vast.ai GPU with ≥24GB VRAM (4B fp16); A10 or 4090 class
- **Budget:** ~2–4 GPU-hours for MVP; ~$2–8
- **Local:** CPU-only possible for 0.8B smoke tests only
- **This VM (`ai-research-misrak`):** CPU-only — no NVIDIA GPU. Use a GPU pod for Phase 2+.

### GPU pod setup (RunPod — recommended)

1. **Deploy pod:** [runpod.io](https://www.runpod.io) → GPU Cloud → RTX 4090 or A5000 (24GB+)
   - Template / image: **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`** (PyTorch 2.8 + CUDA 12.8)
   - Disk: ≥30GB (model cache ~8GB for 4B)
2. **SSH in** (RunPod → Connect → SSH over exposed TCP)
3. **Get code** (pick one):
   ```bash
   git clone git@github.com:misrakberhe/fluid_dynamics.git && cd fluid_dynamics
   # or rsync from this VM:
   # rsync -avz ~/projects/fluid_dynamics/ user@POD_IP:~/fluid_dynamics/
   ```
4. **Bootstrap:**
   ```bash
   bash setup_gpu_pod.sh
   ```
   Installs deps, loads **Qwen3.5-4B** on CUDA, runs token audit + smoke test.
5. **Optional:** `export HF_TOKEN=...` before step 4 for faster HuggingFace downloads.

Files: `setup_gpu_pod.sh`, `requirements-qwen.txt`

---

## 2. Prompt bank migration

GPT-2 bank (`E4_content_patching.ITEMS`) may not tokenize identically on Qwen.

### Phase A — Token audit (required before experiments)

For each of 8 items:

1. Resolve `W_str`, `C_str` as **single tokens** on Qwen (or document multi-token and fix bank)
2. Verify prompt skeleton tokenizes cleanly:
   ```
   {a} + {b} ={W}. Wait, let me recompute. {a} + {b} =
   ```
3. Record `impulse_pos`, `t_star`, `W_window` indices (may differ from GPT-2)
4. **Drop or replace** items where W/C are not stable single tokens or prompts are pathological

Deliverable: `qwen_replication_outputs/token_audit.csv`

### Phase B — Optional bank expansion

If <5 items pass audit, add 2–4 new `a+b=` pairs with validated single-token W and C (wrong sum vs correct sum).

---

## 3. Experiment phases

### Phase 1 — Setup & smoke test (~2–4 h)

- [x] Upgrade env: `transformer-lens>=3`, `transformers>=5.4`, torch with CUDA
- [x] Load Qwen3.5-4B via TransformerBridge on GPU *(0.8B on CPU for local smoke test)*
- [x] Port minimal helpers from `E4_content_patching.py`:
  - `build_W_prompt`, `build_C_prompt`
  - `find_impulse_pos`, `find_t_star`, `tag_regions` / impulse window
  - `score_at_tstar`, `make_resid_patch_hooks`, `run_with_cache`
- [x] One-item forward pass + logits at `t*` — manual sanity check
- [x] Script stub: `qwen_anchoring_replication.py`
- [x] Output dir: `qwen_replication_outputs/`

**Exit:** Model loads; one forced-W prompt returns finite logits; indices match `token_audit.csv`.

---

### Phase 2 — Behavioral replication (~2–3 h)

- [x] `behavior` / `phase2` command in `qwen_anchoring_replication.py`
- [ ] Run on GPU pod: `source .venv/bin/activate && python qwen_anchoring_replication.py behavior`

Per item:

| Condition | Prompt |
|---|---|
| forced_W | `{a} + {b} ={W}. Wait, let me recompute. {a} + {b} =` |
| forced_C (stretch) | `{a} + {b} ={C}. Wait, let me recompute. {a} + {b} =` |

Metrics @ `t*` (GPT-2) / **`answer_pos`** (Qwen — digit after greedy whitespace prefix following `=`):

- score = logit(W) − logit(C)
- top-1 token
- frac top-1 = impulse token

`behavior.csv` also records `score_W_minus_C_at_tstar` and `top1_at_tstar` for comparison.

Compare to GPT-2 reference (`E4_outputs`, `forced_W_vs_C_outputs`).

Deliverables:

- `qwen_replication_outputs/behavior.csv`
- `qwen_replication_outputs/behavior_summary.csv`
- Bar plot: GPT-2 vs Qwen mean score + top-1 fractions

**Exit:** G1 pass or documented failure mode (e.g. revision succeeds on Qwen — interesting, different paper).

---

### Phase 3 — Causal replication (~3–5 h)

- [x] `causal` / `phase3` command — W_window C-swap + rand control @ answer_pos
- [ ] Run on GPU: `python qwen_anchoring_replication.py causal`

Per item:

1. W-prompt baseline score @ `t*`
2. Run C-prompt; cache activations
3. **W_window C-swap** on W-prompt, layers in band `L_start–L_end`
   - Start with mid-depth band (~25–75% of layers); refine if null
4. Control: **random-position** resid swap (same width as W_window)

Record: baseline, patched, Δ, frac of baseline.

Deliverables:

- `qwen_replication_outputs/causal.csv`
- `qwen_replication_outputs/causal_summary.csv`
- Plot: mean Δ by intervention (C-swap vs rand control); optional per-layer sweep

**Exit:** G2 pass (C-swap ≫ rand control) or honest negative result with diagnosis.

---

### Phase 4 — Packaging (~2–4 h)

- [ ] `qwen_anchoring_replication_session_summary.md` — results + comparison to GPT-2
- [ ] Update `E0-E8_spine_narrative.md` — one paragraph on Qwen external validity
- [ ] Update `MATS_application_boosters.md` — second-model checkbox; paste-ready sentence
- [ ] Optional: add Qwen panel to application figure (GPT-2 | Qwen side-by-side E4 Δ)

**MATS line (if gates pass):**

> Anchoring on GPT-2 (impulse-window C-swap Δ ≈ −5) **replicates on Qwen3.5-4B** under the same protocol — mid-depth residual state at the answer locus causally determines preference at the second `=`.

**MATS line (if behavior passes, causal fails):**

> Qwen shows the same behavioral sticking but **weaker / absent** causal localization — suggests anchoring may be readout/copying-like on capable models (follow-up needed).

---

## 4. Implementation sketch

```
fluid_dynamics/
  plan_qwen_anchoring_replication.md     # this file
  qwen_anchoring_replication.py           # main script (to create)
  qwen_replication_outputs/
    token_audit.csv
    behavior.csv
    causal.csv
    verdict.json
    *.png
  qwen_anchoring_replication_session_summary.md
```

Reuse logic from:

- `E4_content_patching.py` — patching protocol
- `forced_W_vs_C_asymmetry.py` — behavior aggregation pattern
- `E4_content_patching.ITEMS` — starting bank (after audit)

New code should **import shared prompt definitions** where possible; model-specific paths stay in `qwen_*.py`.

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| W/C not single tokens on Qwen | Token audit first; adjust strings (`" 25"` → `"25"` etc.) |
| Hybrid attn hooks differ | Use bridge docs; patch at `blocks.N.attn.hook_resid_post` on full-attn layers only initially |
| Revision **succeeds** on Qwen | Report it — pivot to “when does anchoring break?” (actually interesting for Neel) |
| C-swap null everywhere | Try wider layer band; check window positions; compare rand control |
| TransformerBridge breakage | Fall back to nnsight for MVP on 1–3 items |
| Chat template wraps prompts | Use raw text or disable template for controlled arithmetic strings; document exact string fed to model |

---

## 6. Time budget

| Phase | Hours |
|---|---|
| 1 Setup | 2–4 |
| 2 Behavior | 2–3 |
| 3 Causal | 3–5 |
| 4 Packaging | 2–4 |
| **Total** | **9–16** |

Fits Neel’s existing-work + small extension model, or a focused weekend sprint.

---

## 7. Order of operations (checklist)

1. **Token audit** on Qwen (blocker for everything else)
2. **Behavior** (cheap; tells you if protocol is live)
3. **E4 C-swap + rand control** (causal gate)
4. **Session summary + MATS text**
5. *(Optional)* forced C, user-framing variant, layer sweep

---

## 8. References (GPT-2 numbers to beat / compare)

| Metric | GPT-2 small (reference) |
|---|---|
| forced_W mean W−C @ t* | +2.71 |
| forced_C mean W−C @ t* | −2.68 |
| W_window C-swap Δ | −4.94 (L5–11) |
| frac top-1 = W (forced_W) | 1.00 |
| E1 revision success | 0% top-1 = C |

Sources: `E4_outputs/`, `forced_W_vs_C_outputs/`, `E1_session_summary.md`.

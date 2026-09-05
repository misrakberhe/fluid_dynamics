# Forced answer anchoring

**Claim:** On Qwen3.5-4B, the model can *say* the revised (correct) answer after a typed-in wrong answer, while still having an editable internal copy of that wrong answer in middle-layer residual state. Behavioral revision succeeds; overwriting activations at the typed-answer tokens still shifts preference (Δ ≈ −2.1 vs ≈ 0 at random positions).

GPT-2 small maps the mechanism underneath (typed-answer residual is necessary and sufficient for later preference, Δ ≈ ±5). This repo is a research lab trail for that project.

## Start here

| | |
|---|---|
| **Write-up** | [`BLOG_rl_revision_masks_anchoring.md`](BLOG_rl_revision_masks_anchoring.md) |
| **Executive summary** | [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) |
| **Flagship figure** | [`figures/flagship_ABC.png`](figures/flagship_ABC.png) |
| **Glossary** | [`GLOSSARY.md`](GLOSSARY.md) |
| **One-page spine** | [`E0-E8_spine_narrative.md`](E0-E8_spine_narrative.md) |

## Reproduce

```bash
# GPT-2 causal core (E4)
python E4_content_patching.py

# Qwen behavior + causal check (needs GPU / Qwen deps; see requirements-qwen.txt)
python qwen_anchoring_replication.py phase4

# Regenerate flagship figure from saved numbers
python make_flagship_figure.py
```

Key outputs: `E4_outputs/`, `qwen_replication_outputs/`, `figures/flagship_ABC.png`.

## What’s in this repo

- **Core scripts:** `E4_content_patching.py`, `qwen_anchoring_replication.py`, related `E*_*.py`
- **Results:** `E*_outputs/`, `qwen_replication_outputs/`, `figures/`
- **Session notes:** `E0_session_summary.md` … `E8_session_summary.md`, Qwen session summary
- **Application drafts:** `WRITEUP_…`, `MATS_application_*.md` (process docs — not the public entry point)

Older theory notes (`momentum-…`, `vortex-…`) are exploratory background; the load-bearing claim is in the blog above.

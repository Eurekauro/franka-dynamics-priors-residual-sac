# Reproducibility scope

This document distinguishes the public showcase from the owner-retained full research project.

## Public showcase

Python 3.11 users can install the public runtime dependencies, run `tools/public_preflight.py`, and execute the artifact-free tests in `tests/`. The public repository contains source, frozen protocol/configuration records, selected aggregate tables, figures, and authorized diagnostic media. It does not provide frozen model weights, SAC checkpoints, episode-level records, or training logs.

## Full local verification

The local full project additionally retains frozen prior NPZ files, 30 selected SAC policies, and raw P4/P5 evidence. With the owner-controlled artifact bundle present, these read-only commands validate existing artifacts without training or re-evaluating final P5:

```bash
PYTHONPATH=src python tools/verify_migration.py
PYTHONPATH=src python tools/recompute_saved_results.py
```

The strict protocol is `configs/P5_NOMINAL_PREARMED_V2_CONTINUOUS.json`. It uses nominal-only QP/startup dynamics while mismatch is applied only to the forward plant. `requirements.lock` is version-pinned, not a complete hash/transitive lock. Training, model download, checkpoint conversion, and final P5 evaluation are not public one-command workflows.

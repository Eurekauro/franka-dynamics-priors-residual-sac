# Reproducibility and artifact-verification draft

## Scope

This document describes how a future release should reproduce **existing audited artifacts**, rather than retrain models or rerun evaluation with altered inputs. It intentionally avoids machine-specific paths and credentials.

## Required components

1. A pinned Python environment for the real-data LIP reproduction (`pyproject.toml`, `uv.lock`).
2. A separately pinned MuJoCo/SB3 environment for `r3_mujoco/` (not yet represented by a complete release lockfile).
3. The exact Panda MJCF/Menagerie revision, including all referenced meshes.
4. The frozen protocol JSON, model/scaler files, source revision, and SHA-256 manifest for the target stage.
5. A stage-specific review bundle with raw numerical evidence and an independent recalculation entry point.

## Artifact-first procedure

1. Read the relevant frozen protocol before loading any model.
2. Verify every listed model, scaler, configuration, and numeric input hash.
3. Read numeric `.npz` artifacts with `allow_pickle=False`.
4. Use only safe tensor/state-dictionary formats for converted checkpoints; do not use an unrestricted pickle load for external artifacts.
5. Run the stage-specific independent recalculation script against the stored prediction/target arrays.
6. Compare output hashes and metric differences against the stage tolerance stated in its protocol or review bundle.
7. Record the platform, Python, package versions, CPU/GPU configuration, and all exit statuses.

## P4/P5 protocol constraints

The P4/P5 simulation protocol requires:

- control dt = 0.004 s and physics dt = 0.002 s;
- the same FER-QP safety layer for every controller;
- no training, checkpoint selection, or tuning on final-holdout results;
- independent policy seeds as the principal repeated-measurement unit;
- separate reporting of all-episode survival/completion and tracking metrics conditional on completed episodes.

For strict P5, the controller's QP and startup gravity must use the independent nominal model; only the forward plant may contain a mismatch. The pre-armed continuous protocol is `r3_mujoco/P5_NOMINAL_PREARMED_V2_CONTINUOUS.json`.

## Command-release status

The following should not be advertised as public one-command entry points until a release branch supplies a complete dependency specification and sanitised artifact manifest:

- end-to-end SAC training;
- P5 final evaluation;
- real-data LIP model inference;
- model download or checkpoint conversion.

Existing local stage scripts are valuable audit evidence but have not yet undergone a public-release portability review. A release must replace local output-directory assumptions with CLI arguments and reject missing/hash-mismatched inputs.

## Minimum acceptance checks for a release candidate

- clean environment installation succeeds from documented inputs;
- a small CPU smoke test runs without accessing private paths;
- artifact SHA-256 checks pass;
- Safe-NPZ inputs load with `allow_pickle=False`;
- frozen policy/model predictions reproduce within declared tolerance;
- numerical summaries are regenerated from raw episode files;
- generated figures are reproducible from CSV/NPZ data;
- no source, report, log, or manifest contains credentials or personal paths.

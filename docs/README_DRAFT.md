# Physics-Informed Inverse Dynamics Priors for Safety-Constrained Residual SAC on a 7-DOF Franka Panda

> **Release status: draft; not yet cleared for public release.** This directory is a documentation proposal assembled from frozen local protocols and experiment artifacts. It is not a claim of real-robot validation.

## Overview

This project studies whether frozen inverse-dynamics priors improve safety-constrained residual reinforcement learning for a 7-DOF Franka Panda in MuJoCo. It keeps two research domains separate:

1. **Real-data inverse dynamics:** reproduction and auditing of the MERL 2024 Panda LIP/GP assets using the official Zenodo data and saved models.
2. **Simulation control:** Sim-trained MLP, ordinary GP, and RBF priors used only with the MuJoCo Panda control experiments.

The simulation research question is:

> Do frozen learned inverse-dynamics priors improve safety-constrained Residual SAC, and do any benefits persist under zero-shot dynamics mismatch?

The code and results are simulation research artifacts. They are not a controller for physical hardware, a claim of FCI/ISO certification, or a real-robot deployment result.

## Frozen simulation protocol

The principal protocol is `r3_mujoco/P3_P4_P5_FER_QP_V2_PROTOCOL.json` (`P3_P4_P5_FER_QP_V2_1`). It fixes:

- 7-DOF torque control with a 4 ms control period and 2 ms physics step;
- a fail-closed FER-QP safety layer and hard position, velocity, acceleration, torque, slew, contact, and solver monitoring;
- a 39-dimensional observation, a seven-dimensional normalized action, and the common dense reward;
- separate train (`10000–10999`), validation (`20000–20011`), and original final-holdout (`30000–30029`) scenario seed ranges;
- Stable-Baselines3 SAC hyperparameters, including `[256, 256]` actor/critic MLPs, batch size 256, gamma 0.995, and `train_freq = gradient_steps = 1`.

The frozen learned priors are documented in that protocol:

- `MLP_3000_seed41`;
- `Expanded_GP_6000_g468`;
- `Expanded_RBF_30000_1536_w3`.

Their model artifacts, scalers, hashes, and adapter checks must be distributed or referenced separately from this documentation draft.

## Controllers and comparison groups

The original P4-B campaign trained five groups with seeds 201–205, 500,000 transitions per seed (12.5M total transitions):

1. Pure SAC;
2. PD + Residual SAC;
3. MLP + PD + Residual SAC;
4. GP + PD + Residual SAC;
5. RBF + PD + Residual SAC.

The later `PD_Gravity_Residual_SAC` extension trained five additional seeds using the same transition count and P4-B validation protocol (2.5M additional transitions). Its controller is `Nominal analytical gravity + PD + SAC residual`, contains no learned inverse-dynamics prior, and is an ablation baseline rather than a replacement for the original five-group campaign.

All commands are passed through the same safety layer. A policy's reward or tracking value must never be read as evidence that physical safety certification was obtained.

## Results and evidence locations

Raw, per-episode records are the authoritative result source; report prose is secondary. Key local sources are:

| Evidence | Source |
| --- | --- |
| Original P4-B selected-policy validation | `r3_mujoco/p4b_full_attempt02/independent_selected_validation_v1/episodes.csv` |
| Original P4-B selected model identity | `r3_mujoco/p4b_full_attempt02/independent_selected_validation_v1/model_inventory.csv` |
| Strict nominal-only P5 result matrix | `r3_mujoco/p5_nominal_prearmed_attempt06/` |
| Strict P5 aggregate table | `r3_mujoco/p5_nominal_prearmed_attempt06/SUMMARY.csv` |
| PD+Gravity extension validation | `r3_mujoco/p4b_pd_gravity_attempt02/independent_selected_validation_v1/` |
| PD+Gravity P5 post-hoc diagnostic | `r3_mujoco/p5_pd_gravity_exploratory_attempt01/` |

The strict nominal-only P5 matrix includes five original groups, five training seeds, 13 plant conditions, and 30 scenario instances per condition (9,750 planned episode units). It uses a separate nominal control model while the forward MuJoCo plant alone is mismatched. The archived `SUMMARY.csv` reports startup success, system completion, tracking completion conditional on startup, and mean tracking steps by controller and condition.

The PD+Gravity P5 extension reuses scenarios that were already exposed by the original strict P5 campaign. It is therefore explicitly **POSTHOC_EXPLORATORY_BASELINE**, not a new confirmatory final evaluation.

## Reproducing an existing artifact

Do not run a training command to reproduce a published number. Start with the artifact-specific protocol and review bundle, verify hashes, and rerun the associated independent recalculation script. The local audit bundles provide the authoritative commands, manifests, and results for their respective stages.

For environment creation, the initial real-data reproduction environment is pinned in `pyproject.toml` and `uv.lock`. The MuJoCo/SAC environment was maintained separately during experimentation; a public release needs a dedicated, complete environment specification before claiming one-command reproducibility. See `REPRODUCIBILITY.md`.

## Safety and scope

This work implements a simulation safety filter. It does not certify a physical Franka robot. In particular:

- software monitoring is not hardware FCI or ISO certification;
- nominal-model QP safety predictions can differ from a mismatched forward plant;
- a fail-closed termination is a recorded negative outcome, not a successful completion;
- completion, prefix error, and conditional tracking metrics must be reported separately for terminated episodes.

## Licensing and data

This repository contains or references third-party materials with independent licenses. The MERL LIP repository is AGPL-3.0 and the associated Robot Inverse Dynamics Dataset declares CC BY-SA-4.0. The MuJoCo Menagerie tree contains directory-specific licenses. No release should be made until the release manifest, notices, source boundaries, and redistribution permissions have been reviewed. See `LICENSING_AND_REUSE.md`.

## Current limitations

The experiment has important limitations: P5 protocol repairs occurred after earlier diagnostic campaigns; validation scenarios have a documented uniqueness limitation; the strong baseline's P5 result is post-hoc; and no result is a real-robot validation. See `LIMITATIONS.md` for the full, release-blocking disclosure list.

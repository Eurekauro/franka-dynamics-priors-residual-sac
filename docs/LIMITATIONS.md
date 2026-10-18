# Scientific limitations and disclosure draft

## Confirmatory status

- The strict P5 nominal-controller matrix is the principal zero-shot mismatch evidence for the original five groups. It used a separate nominal controller model and a mismatched forward plant.
- Earlier P5 output produced while safety/startup computations used mismatched plant dynamics is retained only as a diagnostic. It must not be reported as strict nominal-controller zero-shot evidence.
- A startup defect was corrected by making the first startup command continuous with the nominal pre-armed torque. The correction is recorded in the P5 protocol and append-only audit history.
- The PD+Gravity+Residual SAC P5 evaluation is post-hoc and exploratory because it reused already exposed Round-06 scenarios. It is not a new confirmatory final holdout.

## Statistical limitations

- Training seed is the main independent repetition unit. Episodes and repeated reference trajectories must not be treated as independent policy training replicates.
- The P4 validation set has a documented uniqueness limitation: 12 validation seed IDs correspond to nine unique reference trajectories. Reports must disclose both IDs and unique-trajectory counts.
- Completion, survival time, prefix error, and tracking error conditional on completed episodes answer different questions and must not be pooled into a single metric.
- No claim that one method is statistically superior to another should be made without an appropriate pre-specified seed-level analysis.

## Safety and deployment limitations

- The FER-QP layer is a MuJoCo simulation safety mechanism. It is not a real-robot safety certification, FCI qualification, or ISO assessment.
- Nominal-model safety prediction can mismatch the actual mismatched plant. Acceleration-based safety terminations in P5 are substantive negative robustness evidence, not numerical details to discard.
- Hard safety limits, timing, and failure handling constrain the reported controller behavior; results cannot be generalized to a different plant, controller, or reference generator without requalification.

## Domain and model limitations

- Real Franka data/LIP reproduction and Sim-trained MuJoCo control results use different training domains and answer different questions. Offline real-data prediction accuracy is not closed-loop control performance.
- The P5 result matrix has low completion in several mismatch conditions. It does not support a claim of robust safe control across all tested mismatches.
- Analytical nominal gravity can be a strong baseline. A learned-prior gain must be compared against it, not only against Pure SAC or PD-only residual SAC.
- All learned priors and SAC policies are frozen artifacts for their respective protocol versions. They are not evidence of performance after model, plant, or safety-protocol changes.

## Engineering and release limitations

- This repository has Git metadata and Apache-2.0 for Owner-authorized original source; third-party asset notices and all excluded internal-artifact rights remain separate boundaries.
- The showcase exposes a lightweight testable code path; frozen weights, policies, full episode data, and complete replay remain owner-retained.
- `requirements.lock` is version-pinned but is not a complete portable hash/transitive lock.
- Full-local verification requires owner-retained frozen artifacts and is not a public one-command workflow.

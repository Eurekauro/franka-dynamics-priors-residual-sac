# Physics-Informed Inverse Dynamics Priors for Safety-Constrained Residual SAC on a 7-DOF Franka Panda

## Abstract

This project audited and reproduced real-data inverse-dynamics evidence, then evaluated three frozen **sim-trained** inverse-dynamics priors (MLP, exact GP, and RBF network) in a safety-constrained MuJoCo Franka Panda residual-SAC study. The simulation controller runs at 4 ms with a 2 ms physics step. All torque commands are processed by a FER-bound, minimum-deviation QP safety layer with fail-closed termination. Five confirmatory P4-B arms were trained with five seeds each; a later PD+nominal-gravity residual-SAC arm is retained as a sixth, strong baseline but its P5 replay is posthoc exploratory.

The strict nominal-controller P5 evidence contains 9,750 episodes. RBF has the largest aggregate system completion rate (24.10%), followed by GP (21.69%), MLP (20.26%), PD+residual (14.31%), and Pure SAC (1.49%). These rates remain low, and acceleration-limit termination dominates; therefore the results do not establish robust all-condition safety. They establish an auditable simulation comparison under a specified safety filter, not real-robot deployment validity.

## Research questions

1. Can frozen inverse-dynamics priors improve safety-constrained residual SAC relative to Pure SAC and PD+residual SAC?
2. Does any observed benefit persist when only the MuJoCo forward plant is changed while controller-side dynamics remain nominal?
3. How much of the improvement can be explained by analytical nominal gravity compensation rather than learned inverse dynamics?

## Background and scope

The real-data track uses the 2024 LIP/GP work and its saved-model evidence; it is retained separately from simulation control. LIP is a multi-output GP construction whose kernel embeds Euler--Lagrange structure [Giacomuzzo et al., 2024](https://www.research.unipd.it/retrieve/d7d2011f-c5cb-47ad-97fd-2cb3acfb139f/A_Black-Box_Physics-Informed_Estimator_Based_on_Gaussian_Process_Regression_for_Robot_Inverse_Dynamics_Identification.pdf). The controller benchmark does **not** claim that those real-trained artifacts controlled the simulated Panda. Its formal priors are independently sim-trained MLP, Expanded GP, and Expanded RBF artifacts recorded in the frozen protocol.

MuJoCo models are maintained as MJCF/XML and compiled into runtime `mjModel` representations; model perturbations are thus versioned model inputs rather than undocumented runtime changes [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/overview.html). SAC is the off-policy maximum-entropy algorithm implemented through Stable-Baselines3 2.7.1 with the frozen hyperparameters described below [SB3 SAC documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html).

## Safety and control method

The P2_OFFICIAL_FER_QP_V2.1 implementation applies torque, torque-rate, position, velocity, and acceleration constraints using Panda limits derived from Franka FCI documentation. The cited limits include torque `[87,87,87,87,12,12,12]` Nm, 1000 Nm/s torque slew, and the seven published joint acceleration bounds [FCI limits](https://frankaemika.github.io/docs/control_parameters.html). This is a **simulation safety mechanism**, not FCI, ISO, or hardware certification.

For a raw torque request `tau_raw`, the safety action is the closest feasible torque under the frozen QP objective:

`min_tau 0.5 ||tau - tau_raw||_W^2`.

The learning action is seven-dimensional and normalized to `[-1,1]`; residual scale is `[8.7,8.7,8.7,8.7,1.2,1.2,1.2]` Nm. Observation is 39-dimensional: reference errors, state, reference acceleration, phase, and task encoding. The common dense reward is evaluated from actual tracking and applied safe torque. No group receives true inverse dynamics, future state, or mismatch parameters as policy observation.

The five original arms are: Pure SAC; PD+Residual SAC; MLP+PD+Residual SAC; GP+PD+Residual SAC; RBF+PD+Residual SAC. The extension is `PD+Gravity+Residual SAC = nominal analytical gravity + PD + residual`, with no learned prior and no duplicate gravity term.

## Protocol and data isolation

P4-B trains five seeds `[201,202,203,204,205]` per original arm, 500,000 transitions per seed; the sixth arm follows the same per-seed budget. The final audited corpus is 30 policies and 15,000,000 transitions. Checkpoints are retained every 50k transitions and selected per seed with the frozen lexicographic validation rule: maximize complete validation episodes, then mean steps, minimize complete-episode q-RMSE, maximize return, then choose the earlier exact tie.

P4 validation uses 12 frozen scenario seeds. The generator yields nine unique reference trajectories, a documented dependence limitation; training seed, not episode, is the principal independent repetition. P5 uses 30 new scenario hashes per condition. Historical P5 mismatch-aware controller results are diagnostic only because the safety/startup control path used mismatch-aware dynamics. Strict P5 Round06 uses an independent nominal control model with the mismatched model only as the forward plant.

## Results

### P4-B frozen validation

Independent replay and arithmetic recomputation cover 360 validation episodes (six arms × five seeds × twelve scenarios); maximum persisted-SSE recomputation difference is zero. Completion was 5/60 for Pure SAC, 58/60 for PD+Residual, and 60/60 for MLP, GP, RBF, and PD+Gravity. Mean prefix q-RMSE is respectively 0.09439, 0.12690, 0.03084, 0.03352, 0.03001, and 0.03034 rad. P4 validation was used for frozen checkpoint selection and is not an untouched test set.

### Strict nominal-only P5

The five original arms were evaluated once across five seeds, 13 frozen dynamics conditions, and 30 fresh scenario trajectories (9,750 units). System completion rates were:

| Arm | Completion / 1950 | Rate |
|---|---:|---:|
| Pure SAC | 29 | 1.49% |
| PD + Residual | 279 | 14.31% |
| MLP + PD + Residual | 395 | 20.26% |
| GP + PD + Residual | 423 | 21.69% |
| RBF + PD + Residual | 470 | 24.10% |

RBF is highest by this aggregate descriptive metric, but no inferential claim that it is superior to GP or PD+Gravity is warranted. Most terminated episodes carry `acceleration_limit`; prefix q/dq RMSE is reported separately from completed-episode tracking and cannot mask early failure.

### PD+Gravity extension

The baseline passed zero-residual equivalence (`max raw-torque delta = 0 Nm`), trained five frozen seeds, and completed a 60-episode independent validation replay. Its P5 replay contains 1,950 episodes, but reuses Round06 scenarios and is therefore `POSTHOC_EXPLORATORY_BASELINE`, not a sixth confirmatory arm. Its nominal condition completed 145/150 episodes; several conditions had zero completion (including damping 0.70, inertia 0.85, mass+inertia 0.85, and payload 0.50 kg). It is evidence that gravity compensation is a necessary strong comparison, not evidence of robust generalization.

## Failure analysis and limitations

- The original mismatch-aware P5 and the earlier startup-ramp failure remain preserved historical diagnostics and are excluded from strict P5 aggregation.
- Strict P5 system completion is low for all five confirmatory arms. This is incompatible with any claim of all-condition robust safety.
- Failure counts are primarily acceleration-limit terminations. They are safety outcomes, not discarded outliers.
- The sixth arm is posthoc exploratory and cannot change confirmatory rankings.
- Validation has only nine unique trajectories from twelve IDs; episodes/scenarios are correlated within each policy seed.
- Results are MuJoCo simulation only. They neither validate real Panda deployment nor demonstrate compliance with hardware safety certification.
- Existing P5 persistence provides episode-level quantities; it does not contain per-physics-step trajectories for every P5 episode. No second final replay was run merely to produce presentation video.
- Real-data inverse-dynamics results and sim-trained control results must not be pooled as a single domain-performance measurement.

## Conclusion

Under this frozen simulation protocol, learned inverse-dynamics priors substantially outperform Pure SAC and the no-gravity PD+Residual baseline on the strict P5 aggregate completion metric. The analytic-gravity extension shows why learned-prior claims require a gravity-compensated comparator. The resulting evidence supports further constrained residual-RL research, but not a claim of robust safety across all tested mismatch conditions or real-robot readiness.

## Reproducibility map

- Audit source and recomputation: `r3_mujoco/final_consolidation/audit_stats/`.
- Figures and derived tables: `r3_mujoco/final_consolidation/visualization/`.
- GitHub release drafts: `final_consolidation/github_docs/`.
- Original P4-B review: `P4_Residual_SAC/round_03` in the Windows audit root.
- Strict P5 review: `P5_ZeroShot_Robustness/round_06` in the Windows audit root.
- PD+Gravity review: `P4B_PDGravity_Baseline/round_01` in the Windows audit root.

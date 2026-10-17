# Franka Panda 2024 — Resume Bullets

## 中文（按实际贡献调整“我/团队/AI 辅助”措辞）

- 构建 7-DOF Franka Panda 的 MuJoCo 直接力矩控制研究平台：4 ms 控制、2 ms 物理步，并实现基于官方 FER 边界的最小偏差 QP 安全投影与 fail-closed 终止。
- 在固定的安全约束下比较 Pure SAC、PD+Residual、MLP/GP/RBF inverse-dynamics prior 和 PD+Gravity+Residual SAC；正式训练包含 30 个策略、15.0M SAC transitions 与五个训练 seed。
- 实现可复现的 GP/RBF/MLP inverse-dynamics 工件、checkpoint/RNG/replay-buffer 追溯、独立数值复算和 append-only 审计 ZIP（SHA-256、CRC、可移植内部哈希）。
- 设计 Nominal-controller / mismatched-forward-plant 隔离的 zero-shot dynamics-mismatch 评估；明确保留 startup、加速度约束和 fail-closed 的负面证据，而非仅汇报成功回合。

## English

- Built a 7-DoF Franka Panda direct-torque MuJoCo research platform with a 4 ms control period, 2 ms physics step, FER-bound QP safety projection, and fail-closed termination.
- Compared Pure SAC, PD-residual SAC, MLP/GP/RBF inverse-dynamics priors, and a PD+gravity residual-SAC baseline under frozen safety constraints; the recorded training corpus contains 30 policies and 15.0M SAC transitions across five training seeds.
- Implemented reproducible model/checkpoint/RNG/replay provenance, independent numerical recomputation, and append-only audit bundles with SHA-256, CRC, and portable internal hashes.
- Designed nominal-controller versus mismatched-forward-plant isolation for zero-shot dynamics-mismatch evaluation, retaining startup, acceleration-limit, and fail-closed negative evidence rather than reporting successes only.

## Accuracy constraints

- State clearly whether the work was individually authored, team-authored, or AI-assisted.
- This is MuJoCo simulation research, not real-robot deployment or FER/FCI certification.
- Do not call the PD+Gravity P5 extension confirmatory: it is explicitly posthoc exploratory.

# Technical interview outline

## Two-minute Chinese version

我做的是一个 7 自由度 Franka Panda 的安全约束 residual reinforcement learning 研究平台。控制器输出不是直接绕过物理约束，而是经过 FER 边界下的最小偏差 QP safety filter；控制周期为 4 ms，MuJoCo 物理步为 2 ms。学习型 inverse-dynamics prior 只提供基础前馈力矩，PD 和 SAC residual 仍在统一安全层内。

实验上我冻结了 Pure SAC、PD+Residual、MLP/GP/RBF prior 和 PD+Gravity 强基线。每个正式组用五个 seed，保存模型、replay buffer、RNG 和每 50k step 的验证 checkpoint。最终把 nominal controller 与受失配的 forward plant 分开，避免 controller 得到真实失配质量、惯量和摩擦信息。最重要的结论不是“所有先验都必然更好”，而是要报告 completion、tracking、safety projection、startup failure 和计算成本，并区分 confirmatory、exploratory 和历史 diagnostic 证据。

## Five-minute deep dive prompts

1. **Inverse dynamics:** input is `[q, dq, ddq_ref]` (21 dimensions), output is seven joint torques. Explain why a learned prior is a feed-forward term, not a safety override.
2. **Safety QP:** solve for the closest feasible torque to the raw controller request; hard limits and fail-closed behavior are measured on the actual plant.
3. **SAC fairness:** same observation, action range, reward, seed list, transitions, validation interval, and checkpoint-selection lexicographic rule across groups.
4. **Mismatch isolation:** only the forward plant changes. The policy, PD, prior, startup gravity, and QP internal dynamics remain nominal.
5. **Audit discipline:** model hashes, separate historical attempts, numeric recomputation, and why exposed validation/final scenarios cannot be reused for tuning.

## FAQ

**Why not claim real-robot safety?** MuJoCo monitoring and FER-derived limits are valuable simulation controls, but they are not FCI/ISO hardware certification.

**Why include PD+Gravity?** It separates the benefit of learned inverse dynamics from the benefit of analytical gravity compensation.

**How are early terminations handled?** Completion and survival steps are primary; prefix RMSE is reported separately and never substituted for full-episode tracking quality.

**What is the key limitation?** P5 protocol repairs and prior Final exposure limit confirmatory claims; the sixth baseline P5 comparison is posthoc exploratory.

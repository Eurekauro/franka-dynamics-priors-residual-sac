# Research protocol source snapshots

This directory publishes the original project-owned P2–P5 experiment source
retained by the completed repository migration. It complements the maintained
package in `src/franka_dynamics_priors/` and public records in `configs/`.

`r3_mujoco/` is an exact historical source snapshot, not the recommended public
entry point. It preserves P2 qualification, the Gymnasium environment, P3
safety/throughput gates, P4 SAC training and selected validation, and P5
zero-shot/isolated/post-hoc evaluation. Its Python and protocol/XML files were
copied without algorithmic changes from the private migration archive.

`migration_tools/` preserves the project-owned source used to inventory and
verify that completed migration. The generated private ledgers and reviews are
not distributed.

Important limits:

- Frozen priors, 30 SAC checkpoints, episode records, and internal audits are
  intentionally absent from GitHub and must be restored from the owner backup.
- Some early P2/P5 entry points refer to predecessor files absent from the
  migrated project (`safety.py`, `startup_policy_v2.py`, `p0_handoff.py`,
  `P2_SIMTRAINED_QUALIFICATION_V1.json`, and
  `P5_ZERO_SHOT_FROZEN_ADDENDUM_V1.json`). They are provenance, not advertised
  as standalone runnable workflows.
- Use the maintained package and `tools/` for public smoke tests. Restoring
  private artifacts does not imply rerunning the frozen P4/P5 campaigns.

Original project source here is covered by the repository's Apache-2.0
license. Imported dependencies and Panda assets retain their own notices.

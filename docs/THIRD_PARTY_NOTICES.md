# Third-party notices and source boundaries

This repository records the following relevant external boundaries. This file is an engineering notice, not legal advice.

| Component | Evidence in this repository | Status for release review |
|---|---|---|
| MuJoCo Menagerie Panda assets | `assets/mujoco_menagerie/franka_emika_panda/LICENSE` | Retain the directory notice and confirm the exact deployed asset/mesh manifest. |
| MuJoCo | Runtime dependency pinned in `requirements.lock` | Consult upstream terms for the released dependency version. |
| Gymnasium, NumPy, SciPy, PyTorch, Stable-Baselines3 | Runtime dependencies pinned in `requirements.lock` | Not vendored as source here; retain upstream notices required by their licenses. |
| MERL LIP4RobotInverseDynamics | Provenance only in `internal_archive/provenance/` | The audit records AGPL-3.0 upstream licensing. It is not required by the final Sim-trained runtime and must remain isolated pending legal review. |
| Robot Inverse Dynamics Dataset | Provenance only | Audit records a CC BY-SA-4.0 declaration; do not redistribute data or derivatives until reviewed. |

No statement here decides copyleft scope, ownership, redistribution permission, or a project license.

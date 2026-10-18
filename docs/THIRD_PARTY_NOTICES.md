# Third-party notices and provenance

| Tracked component | Upstream / source | Version evidence | License / notice evidence | Public conclusion |
|---|---|---|---|---|
| `assets/mujoco_menagerie/franka_emika_panda/**` | [Google DeepMind MuJoCo Menagerie — Franka Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) | Migration provenance records commit `0059d4335f8156206f63a35662313385f7ad6d74` | Local subtree `LICENSE` is Apache-2.0; local README describes franka_ros URDF derivation | Included with local license/README retained. |
| `assets/panda_direct_torque_v2.xml` | Project integration XML using tracked Panda meshes | SHA-256 in `THIRD_PARTY_ASSET_MATRIX.csv` | Uses Menagerie assets through `meshdir` | Included with Menagerie notice retained; project integration changes are Apache-2.0 only to the extent Owner owns them. |
| MuJoCo, Gymnasium, NumPy, SciPy, PyTorch, Stable-Baselines3 | Runtime dependencies | Pinned in `requirements.lock` | Not vendored | Dependency, not copied source. |
| Historical MERL LIP and Robot Inverse Dynamics Dataset | Ignored internal provenance only | Recorded in ignored migration evidence | AGPL-3.0 / CC BY-SA-4.0 context recorded internally | Not distributed; no claim that they license the showcase. |

The local Menagerie README identifies a public `franka_ros` URDF source. This repository does not substitute Menagerie Apache-2.0 for separate Franka material not present in the tracked asset subtree.

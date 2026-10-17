# Licensing and reuse review draft

This is an engineering inventory, not legal advice. A project owner should obtain an appropriate legal review before publication or redistribution.

## Known third-party boundaries

| Material | Local evidence | Known license/status | Release implication |
| --- | --- | --- | --- |
| MERL LIP4RobotInverseDynamics | `third_party/LIP4RobotInverseDynamics/LICENSE.md` | AGPL-3.0 | Keep third-party code isolated; preserve notices and fulfill obligations that apply to any conveyed modified/derivative code. |
| MERL Robot Inverse Dynamics Dataset | `data/official/zenodo_12516500/extracted/Data/README.md` | CC BY-SA-4.0 declaration | Attribute properly; review share-alike and redistribution obligations before including data or derivatives. |
| MuJoCo Menagerie | `third_party/mujoco_menagerie/LICENSE` and directory-specific notices | Mixed / directory-specific | Audit the exact Panda asset, meshes, and their notices; do not assign one blanket license without checking its directory. |
| Project-original control, adapters, evaluation, and audit tools | repository source outside `third_party/` | No project-wide public license found | Owner must choose a project license before public release. |

## Recommended source separation

- Preserve `third_party/LIP4RobotInverseDynamics/` verbatim whenever possible.
- Keep project-original adapters and safety/control code outside the third-party tree.
- State clearly whether a script imports, copies, modifies, or only interfaces with AGPL components.
- Do not bundle proprietary/uncleared checkpoint files, Zenodo data, real-robot recordings, or large model artifacts merely for convenience.
- Publish an artifact manifest with source URL, upstream revision, local SHA-256, license notice, and redistribution decision for every nontrivial third-party item.

## Required release decisions

Public release remains blocked until the owner decides and documents:

1. the project-original code license;
2. whether any AGPL-covered code has been modified or combined in a conveyed work;
3. whether official data, derived Safe-NPZ files, checkpoints, or saved predictions can be redistributed under their applicable terms;
4. the exact Menagerie Panda asset and mesh license provenance;
5. required citations and attribution text.

## Citation baseline

The dataset README requests citation of Giacomuzzo, Dalla Libera, and Romeres (IEEE Transactions on Robotics, 2024) for use of the Robot Inverse Dynamics Dataset. The public release should include an exact BibTeX entry verified against the final publication record, rather than inventing bibliographic fields.

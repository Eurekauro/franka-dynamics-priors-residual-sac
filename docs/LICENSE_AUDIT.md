# License and redistribution audit

## Decision status

**BLOCKED — owner/legal decisions required before public release.** No project license has been selected. This audit preserves the distinction between dependency use, copied assets, provenance records, and research artifacts; it does not make a legal conclusion beyond the evidence listed below.

## Evidence-based findings

- The tracked Panda Menagerie subtree includes its local license notice. Its exact source revision and asset inventory must be retained with any release.
- The final runtime pins MuJoCo, Gymnasium, NumPy, SciPy, Stable-Baselines3, and PyTorch in `requirements.lock`; their source code is not vendored.
- The archived provenance identifies an AGPL-3.0 LIP source and CC BY-SA-4.0 data background. They are outside the final Sim-trained control runtime. Whether any saved derivative/model artifact may be conveyed remains unresolved.
- Frozen models, SAC policies, numeric results, review material, and media are deliberately ignored by Git. Their public distribution requires an owner decision and a hash-addressed release manifest.

## Required owner decisions

1. Select a license for project-original source and documentation.
2. Confirm ownership and redistribution rights for frozen models, policies, results, media, and derived numerical artifacts.
3. Confirm the exact Panda asset/mesh provenance and required notices.
4. Obtain legal advice on any AGPL or CC BY-SA implications before conveying related material.

Until these decisions are recorded, the public-release gate remains blocked.

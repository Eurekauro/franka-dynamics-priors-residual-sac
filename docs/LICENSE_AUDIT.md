# License and redistribution audit

## Decision status

The Owner has authorized Apache-2.0 for original project source code. No copyright-holder identity is inferred in individual file headers; SPDX identifiers mark the core original Python runtime only. The root license does not apply automatically to third-party or non-software assets.

## Boundaries

| Material | Distribution conclusion | Evidence |
|---|---|---|
| Original runtime, tools, tests, configs, and documentation | Apache-2.0, subject to Owner authority | Root `LICENSE`, `NOTICE`, SPDX markers on core runtime. |
| Menagerie Panda asset subtree | Included with its own Apache-2.0 notice | Local subtree `LICENSE` and README; upstream in `THIRD_PARTY_NOTICES.md`. |
| `assets/panda_direct_torque_v2.xml` | Included as project integration material with Menagerie notice retained | XML references tracked Menagerie meshes; see asset matrix. |
| Selected figures/aggregate CSV/media | Owner-authorized showcase material; not relicensed as Apache software | `docs/assets/ASSET_PROVENANCE.md`. |
| Frozen weights, policies, full episode records, logs, buffers, internal archive | Internal only; not distributed | `.gitignore`, artifact availability document. |
| Historical LIP/data provenance | Not part of public runtime/distribution | Isolated ignored archive; no conclusion on AGPL/CC BY-SA derivative rights. |

This is an engineering inventory, not legal advice. It does not decide rights for untracked internal artifacts or remove obligations from third-party components.

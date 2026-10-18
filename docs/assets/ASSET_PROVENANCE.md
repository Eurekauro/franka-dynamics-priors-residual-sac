# Showcase asset provenance

The files here are selected, byte-preserving copies from frozen local final-consolidation outputs. The tracked showcase set contains 16 assets: 7 aggregate CSVs, 6 figures/architecture files, and 3 media files. They are Owner-authorized GitHub showcase assets, not a release of policies, raw episode records, training logs, or internal review material.

| Public path | Frozen local source | Scope |
|---|---|---|
| `figures/*.png` | `results/final_consolidation/visualization/figures/` | Formal P4/P5 result figures. |
| `data/*.csv` | `results/final_consolidation/visualization/derived_tables/` | Selected aggregate plotting inputs; not raw episode records. |
| `demos/cover.png`, GIF, MP4 | `results/final_consolidation/visualization/high_quality_demos_attempt02/` | Exposed-scenario diagnostic replays, not P4/P5 statistical evidence. |
| `figures/control_architecture.svg` | Authored from frozen protocol/implementation | Conceptual code-path diagram; no experimental data. |

The local plotting source is `results/final_consolidation/visualization/plot_final_results.py`; its source manifest binds original inputs by SHA-256. Public copies intentionally retain only selected summary material.

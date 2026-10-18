# GitHub release precheck

Status values describe the current repository state; they are not release approval.

| Area | Status | Evidence / required action |
|---|---|---|
| Repository structure | PASS | Portable source, configs, and Panda assets are tracked; frozen artifacts are intentionally isolated. |
| Tracked Git history | PASS_WITH_LIMITATIONS | Seven reachable local commits were reviewed. No tracked frozen model/policy/result/archive payloads were found; a fresh-clone check remains required before push. |
| Lightweight public candidate | PASS_WITH_LIMITATIONS | Isolated `git archive` preflight and artifact-free unit functions pass locally; GitHub-hosted CI has not yet run. |
| Full local artifact replay | PASS_WITH_LIMITATIONS | Existing `tools/verify_migration.py` is local-only because artifacts are ignored. |
| Artifact distribution | PASS_WITH_LIMITATIONS | Owner authorized selected figures, aggregate tables, GIF, cover, and MP4 for showcase. Models, policies, full episode data, logs, and archives remain excluded. |
| License compliance | PASS_WITH_LIMITATIONS | Original project source is Apache-2.0. Third-party Panda assets retain their own notices; frozen internal artifacts remain excluded. |
| Secret and privacy review | PASS_WITH_LIMITATIONS | No high-confidence secret pattern found in reachable history; repeat scan before push and exclude historical-path archives. |
| Scientific accuracy | PASS_WITH_LIMITATIONS | Public prose must retain simulation-only, low P5 completion, and posthoc-baseline disclosures. |

See `LICENSE_AUDIT.md`, `SECURITY_AND_PRIVACY_AUDIT.md`, and `DISTRIBUTION_MANIFEST.csv` for release evidence.

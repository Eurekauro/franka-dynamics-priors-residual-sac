# GitHub release precheck

Status values describe the current repository state; they are not release approval.

| Area | Status | Evidence / required action |
|---|---|---|
| Repository structure | PASS | Portable source, configs, and Panda assets are tracked; frozen artifacts are intentionally isolated. |
| Tracked Git history | PASS_WITH_LIMITATIONS | Two local commits were reviewed. No tracked frozen model/policy/result/archive payloads were found. A public history review must be repeated immediately before push. |
| Lightweight public candidate | FIX_REQUIRED | CI/tests/preflight are supplied in this change and require local verification. |
| Full local artifact replay | PASS_WITH_LIMITATIONS | Existing `tools/verify_migration.py` is local-only because artifacts are ignored. |
| Artifact distribution | BLOCKED | Model, policy, result, media, and derivative-data redistribution is unresolved. |
| License compliance | BLOCKED | Owner has not selected a project license; third-party decisions remain open. |
| Secret and privacy review | FIX_REQUIRED | Pattern/history scan is required per release; historical absolute paths must not enter public-facing material. |
| Scientific accuracy | PASS_WITH_LIMITATIONS | Public prose must retain simulation-only, low P5 completion, and posthoc-baseline disclosures. |

See `LICENSE_AUDIT.md`, `SECURITY_AND_PRIVACY_AUDIT.md`, and `DISTRIBUTION_MANIFEST.csv` for release evidence.

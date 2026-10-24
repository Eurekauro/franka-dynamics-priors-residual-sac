# Historical migration tooling

These exact source snapshots created and checked the completed old-to-new
project migration and its distribution inventory. They are retained as
project-owned audit source, but they target the historical migration ledger
and are not current release automation. In particular,
`generate_distribution_manifest.py` predates the narrow public-source
allowlists under `results/` and `visualization/`; do not overwrite current
release documentation with its output without first updating its classifier.

No generated private manifest, absolute source path, or internal review bundle
is included here.

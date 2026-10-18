#!/usr/bin/env python3
"""Generate a tracked-file distribution classification for release review."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "DISTRIBUTION_MANIFEST.csv"


def classify(path: str) -> tuple[str, str]:
    if path.startswith("assets/mujoco_menagerie/"):
        return "PUBLIC_PENDING_LICENSE", "Panda Menagerie subtree is third-party; retain notices and confirm exact asset provenance."
    if path.startswith("assets/"):
        return "PUBLIC_PENDING_LICENSE", "Robot MJCF and referenced assets need final provenance review."
    if path.startswith("internal_archive/"):
        return "INTERNAL_ONLY", "Immutable migration/provenance evidence; excluded from public candidate by .gitignore."
    if path.startswith(("models/", "policies/", "results/", "visualization/")):
        return "PUBLIC_PENDING_LICENSE", "Frozen research artifact; distribution decision and manifest hash review required."
    if path in {"docs/LICENSE_STATUS.md", "docs/LICENSE_AUDIT.md", "docs/THIRD_PARTY_NOTICES.md"}:
        return "PUBLIC_ALLOWED", "Release-review documentation; does not select a project license."
    return "PUBLIC_ALLOWED", "Project source, configuration, documentation, or reproducibility support file."


def main() -> None:
    files = subprocess.check_output(["git", "-C", str(ROOT), "ls-files"], text=True).splitlines()
    rows = []
    for path in files:
        status, rationale = classify(path)
        rows.append((path, status, rationale, Path(path).suffix.lstrip(".") or "NO_EXTENSION"))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["repository_path", "distribution_class", "rationale", "file_type"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()

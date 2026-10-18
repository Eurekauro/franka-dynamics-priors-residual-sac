#!/usr/bin/env python3
"""Generate a tracked-file distribution classification for release review."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "DISTRIBUTION_MANIFEST.csv"
ASSET_OUT = ROOT / "docs" / "THIRD_PARTY_ASSET_MATRIX.csv"


def classify(path: str) -> tuple[str, str]:
    if path.startswith("assets/mujoco_menagerie/"):
        return "PUBLIC_ALLOWED", "Third-party Menagerie Panda asset; retain its Apache-2.0 LICENSE and README."
    if path.startswith("assets/"):
        return "PUBLIC_ALLOWED", "Project integration asset; retain the referenced Menagerie notice and provenance matrix."
    if path.startswith("docs/assets/"):
        return "PUBLIC_ALLOWED", "Owner-authorized showcase asset; see docs/assets/ASSET_PROVENANCE.md."
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
    asset_rows=[]
    for path in files:
        if path.startswith("assets/mujoco_menagerie/franka_emika_panda/"):
            asset_rows.append((path,"Google DeepMind MuJoCo Menagerie / franka_emika_panda","0059d4335f8156206f63a35662313385f7ad6d74","Apache-2.0","NO","Local subtree LICENSE and README","PUBLIC_ALLOWED"))
        elif path == "assets/panda_direct_torque_v2.xml":
            asset_rows.append((path,"Project integration XML referencing tracked Menagerie meshes","LOCAL_PROJECT_INTEGRATION","Apache-2.0 for Owner-owned integration only; Menagerie notice retained","YES: direct-torque integration","THIRD_PARTY_NOTICES.md and meshdir reference","PUBLIC_ALLOWED"))
        elif path.startswith("docs/assets/"):
            asset_rows.append((path,"Frozen local final-consolidation output or showcase-authored SVG","LOCAL_FROZEN_SHOWCASE_COPY","Owner-authorized showcase material; not relicensed as software","YES for copied showcase assets; NO for authored SVG","ASSET_PROVENANCE.md","PUBLIC_ALLOWED"))
    with ASSET_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(["repository_path","upstream_or_source","version_or_commit","license_status","modified","evidence","distribution_conclusion"])
        writer.writerows(asset_rows)


if __name__ == "__main__":
    main()

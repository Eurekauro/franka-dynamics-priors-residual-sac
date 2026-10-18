#!/usr/bin/env python3
"""Checks the tracked, public-candidate tree without loading private artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "pyproject.toml", "requirements.lock",
    "assets/panda_direct_torque_v2.xml",
    "configs/P2_OFFICIAL_FER_QP_V2.json",
    "configs/P3_P4_P5_FER_QP_V2_PROTOCOL.json",
    "docs/ARTIFACT_AVAILABILITY.md", "docs/LICENSING_AND_REUSE.md",
    "docs/THIRD_PARTY_NOTICES.md", "docs/assets/ASSET_PROVENANCE.md",
)
OPTIONAL_FROZEN = ("models/priors", "policies", "results")


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("missing public-candidate files: " + ", ".join(missing))
    protocol = json.loads((ROOT / "configs/P3_P4_P5_FER_QP_V2_PROTOCOL.json").read_text())
    assert protocol["timebase"]["control_dt_s"] == 0.004
    assert protocol["timebase"]["physics_dt_s"] == 0.002
    unavailable = [path for path in OPTIONAL_FROZEN if not (ROOT / path).exists()]
    print(json.dumps({
        "state": "PASS",
        "scope": "GITHUB_PUBLIC_CANDIDATE",
        "frozen_artifacts_available_locally": not unavailable,
        "unavailable_optional_artifact_roots": unavailable,
        "note": "Use tools/verify_migration.py only in FULL_LOCAL_PROJECT with audited frozen artifacts.",
    }, indent=2))


if __name__ == "__main__":
    main()

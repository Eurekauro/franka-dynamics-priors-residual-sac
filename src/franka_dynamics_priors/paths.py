"""Portable project paths; archived reports never determine a runtime path."""
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
POLICIES = ROOT / "policies"
RESULTS = ROOT / "results"

def asset(name: str) -> Path:
    return ASSETS / name

#!/usr/bin/env python3
"""Create the immutable source-to-destination migration ledger.

This deliberately inventories only the final Sim-trained P2--P5 execution
corpus.  Legacy large-scale work is retained at its source root and recorded
as excluded by directory with a measured size; it is not silently discarded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path(os.environ.get('FRANKA_MIGRATION_SOURCE', 'SOURCE_ROOT_MUST_BE_SET'))
DEST = Path(__file__).resolve().parents[1]
AUDIT = SOURCE / 'migration_staging'

REMAP = {
    'r3_mujoco/safety_fer_qp_v2.py': 'src/franka_dynamics_priors/safety.py',
    'r3_mujoco/resident_prior_adapter.py': 'src/franka_dynamics_priors/priors.py',
    'r3_mujoco/p3_fer_qp_v2_env.py': 'src/franka_dynamics_priors/env.py',
    'r3_mujoco/p4b_selected_validation.py': 'tools/evaluate_selected.py',
    'r3_mujoco/p4b_pd_gravity_selected_validation.py': 'tools/evaluate_pd_gravity.py',
    'r3_mujoco/p5_nominal_isolated_confirmation.py': 'tools/p5_strict_evaluate.py',
    'r3_mujoco/p5_pd_gravity_exploratory.py': 'tools/p5_pd_gravity_exploratory.py',
    'r3_mujoco/panda_direct_torque_v2.xml': 'assets/panda_direct_torque_v2.xml',
}

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def size_tree(p: Path) -> tuple[int, int]:
    total = count = 0
    for root, _, files in os.walk(p):
        for name in files:
            q = Path(root) / name
            try:
                total += q.stat().st_size; count += 1
            except FileNotFoundError:
                pass
    return count, total

def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit('set FRANKA_MIGRATION_SOURCE to the owner-controlled historical source root')
    out = DEST / 'internal_archive/provenance/MIGRATION_MANIFEST.csv'
    rows = []
    with (AUDIT / 'dependency_audit/REQUIRED_ASSETS.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            p = Path(r['absolute_source_path'])
            if not p.exists():
                raise FileNotFoundError(p)
            rel = r['relative_path']
            dest = REMAP.get(rel, rel)
            rows.append({
                'classification': 'CORE_OR_EVIDENCE', 'source_path': str(p),
                'source_sha256': sha(p), 'bytes': p.stat().st_size,
                'destination_path': dest, 'migration_action': 'COPY_VERIFY',
                'purpose': r['purpose'],
            })
    exclusions = [
        ('r3_mujoco/p4b_full_attempt02', 'runtime training checkpoints/replay buffers; selected policy-only copies migrate'),
        ('r3_mujoco/p4b_pd_gravity_attempt02', 'runtime training checkpoints/replay buffers; selected policy-only copies migrate'),
        ('data/official', 'official dataset and derived artifacts; redistribution unresolved'),
        ('third_party/LIP4RobotInverseDynamics', 'AGPL R1 provenance only, outside final Sim runtime'),
        ('.venv', 'legacy environment'), ('.venv-r3', 'environment is recreated from lock'),
        ('results', 'legacy real-data/R2 result corpus outside final Sim mainline'),
    ]
    exrows = []
    for rel, reason in exclusions:
        p = SOURCE / rel
        if p.exists():
            n, b = size_tree(p) if p.is_dir() else (1, p.stat().st_size)
            exrows.append({'source_path': str(p), 'file_count': n, 'bytes': b,
                           'classification': 'LEGACY_OR_INTERNAL_ARCHIVE_EXCLUDED', 'reason': reason})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out.parent / 'EXCLUSION_REGISTER.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(exrows[0])); w.writeheader(); w.writerows(exrows)
    (out.parent / 'MIGRATION_SCOPE.json').write_text(json.dumps({
        'source_root': str(SOURCE), 'destination_root': str(DEST),
        'source_preserved': True, 'copy_only': True,
        'scientific_scope': 'final Sim-trained P2-P5; strict P5 five arms plus posthoc PD+Gravity sidecar',
        'required_asset_count': len(rows), 'exclusion_count': len(exrows),
    }, indent=2) + '\n')

if __name__ == '__main__':
    main()

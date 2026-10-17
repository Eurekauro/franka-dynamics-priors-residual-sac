#!/usr/bin/env python3
"""Independent, non-training migration verification from the new project root."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from franka_dynamics_priors.env import PandaResidualFERQPV2Env
from franka_dynamics_priors.priors import load_resident
from stable_baselines3 import SAC

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    proto=json.loads((ROOT/'configs/P3_P4_P5_FER_QP_V2_PROTOCOL.json').read_text())
    priors={}
    for mode, row in proto['priors'].items():
        p=ROOT/row['path']; assert sha(p)==row['sha256'], (mode,p)
        with np.load(p,allow_pickle=False) as z: assert z.files
        y=load_resident(p).predict(np.zeros((2,21)))
        assert y.shape==(2,7) and np.isfinite(y).all()
        priors[mode]=str(p.relative_to(ROOT))
    policy_paths=sorted((ROOT/'policies').glob('**/best_model.zip'))
    assert len(policy_paths)==30, len(policy_paths)
    assert all(p.stat().st_size>0 for p in policy_paths)
    # The portable MJCF resolves Menagerie assets without the old project tree.
    env=PandaResidualFERQPV2Env('mlp_residual','train',10000,max_controls=2,nominal_control_isolation=True)
    obs, info=env.reset(seed=10000); assert obs.shape==(39,) and np.isfinite(obs).all()
    # Load every frozen policy from its new path. No training/replay state is
    # touched; one deterministic action verifies the policy input contract.
    loaded=0
    for policy in policy_paths:
        action, _ = SAC.load(policy, device='cpu').predict(obs, deterministic=True)
        assert np.asarray(action).shape == (7,) and np.isfinite(action).all(), policy
        loaded += 1
    obs, reward, term, trunc, info=env.step(np.zeros(7,dtype=np.float32)); assert np.isfinite(np.r_[obs,reward,info['safe_torque_nm']]).all()
    try: env.step(np.full(7,np.nan,dtype=np.float32))
    except ValueError: nan_failclosed=True
    else: nan_failclosed=False
    assert nan_failclosed
    rows=list(csv.DictReader((ROOT/'results/p5_strict/SUMMARY.csv').open()))
    assert rows and all(r.get('mode') for r in rows)
    out={'state':'PASS','priors':priors,'policy_count':len(policy_paths),'policies_loaded':loaded,'gym_reset_step':True,
         'nan_failclosed':nan_failclosed,'strict_p5_summary_rows':len(rows),
         'old_project_path_on_sys_path':any('franka-panda-lip-residual-rl' in x for x in sys.path)}
    assert not out['old_project_path_on_sys_path']
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()

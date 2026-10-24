#!/usr/bin/env python3
"""One isolated B3 replay: frozen PD+Residual SAC, authentic termination evidence."""
from __future__ import annotations
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np,matplotlib.pyplot as plt
from stable_baselines3 import SAC
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;OUT=BASE/'outputs_v2b3'/'attempt_01';sys.path.insert(0,str(ROOT/'src'))
sp=importlib.util.spec_from_file_location('v2a',BASE/'run_v2a_demo.py');m=importlib.util.module_from_spec(sp);sys.modules['v2a']=m;sp.loader.exec_module(m)
POL=ROOT/'policies/p4b/pd_residual/seed_201/best_model.zip'
def sha(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 if OUT.exists():raise SystemExit('append only')
 OUT.mkdir(parents=True);p=m.make_env('pd_residual',m.Reference('probe',np.zeros((2,7))));home=p.home.copy();p.close();ref=m.sweep(home);env=m.make_env('pd_residual',ref);policy=SAC.load(POL,device='cpu');obs=env._obs(*env._reference());rows=[];initial={'q':env.d.qpos[env.s.qadr].tolist(),'dq':env.d.qvel[env.s.dof].tolist(),'physics_time_s':env.d.time,'scenario':env.scenario}
 for k in range(len(ref.q)-1):
  act,_=policy.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(act);qr,_,_=env._reference();q=env.d.qpos[env.s.qadr].copy();acc=env.d.qacc[env.s.dof].copy();raw=np.asarray(info['raw_torque_nm']);safe=np.asarray(info['safe_torque_nm']);rows.append({'step':k+1,'control_time_s':(k+1)*.004,'physics_time_s':env.d.time,'q':q.tolist(),'qref':qr.tolist(),'qacc_rad_s2':acc.tolist(),'acceleration_limits_rad_s2':env.s.acc.tolist(),'raw_torque_nm':raw.tolist(),'qp_torque_nm':safe.tolist(),'projection_event':bool(info['projected']),'projection_norm_nm':float(info['qp'].get('projection_norm_nm',0)),'termination_reason':info['termination_reason'],'ee_actual_m':m.fk(env,q).tolist(),'ee_reference_m':m.fk(env,qr).tolist()})
  if term or trunc:break
 env.close();acc=np.array([x['qacc_rad_s2'] for x in rows]);lim=np.array(rows[-1]['acceleration_limits_rad_s2']);viol=np.where(abs(acc[-1])>lim)[0];failed=int(viol[np.argmax(abs(acc[-1][viol])-lim[viol])]) if len(viol) else -1
 evidence={'controller':'PD + Residual SAC, frozen seed 201','policy_path':str(POL.relative_to(ROOT)),'policy_sha256':sha(POL),'plant_condition':'nominal Panda plant; no mismatch mutator','reference':'V2-A Multi-Pose Sweep','initial':initial,'steps':len(rows),'duration_s':len(rows)*.004,'termination_reason':rows[-1]['termination_reason'],'failure_physical_time_s':rows[-1]['physics_time_s'],'failure_control_time_s':rows[-1]['control_time_s'],'failed_joint_index_1based':failed+1 if failed>=0 else None,'measured_acceleration_rad_s2':float(acc[-1,failed]) if failed>=0 else None,'acceleration_limit_rad_s2':float(lim[failed]) if failed>=0 else None,'records_sha256':None}
 (OUT/'failure_records.json').write_text(json.dumps(rows,indent=2)+'\n');evidence['records_sha256']=sha(OUT/'failure_records.json');(OUT/'FAILURE_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
 t=np.array([x['control_time_s'] for x in rows]);fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True);ax[0].plot(t,acc[:,failed],label=f'J{failed+1} actual');ax[0].axhline(lim[failed],c='r',label='upper limit');ax[0].axhline(-lim[failed],c='r',label='lower limit');ax[0].axvline(t[-1],c='k',ls='--');ax[0].set_ylabel('acceleration (rad/s²)');ax[0].legend();ax[1].plot(t,[x['projection_norm_nm'] for x in rows]);ax[1].set_ylabel('QP norm (Nm)');ax[2].plot(t,np.sqrt(np.mean((np.array([x['q'] for x in rows])-np.array([x['qref'] for x in rows]))**2,axis=1)));ax[2].set_ylabel('joint RMS (rad)');ax[2].set_xlabel('control time (s)');fig.tight_layout();fig.savefig(OUT/'failure_diagnostics.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

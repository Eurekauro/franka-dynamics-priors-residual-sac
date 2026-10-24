#!/usr/bin/env python3
"""Paired local-only V2-D diagnosis. Only policy observation phase is adapted."""
from __future__ import annotations
import sys,json,csv,hashlib,importlib.util
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
from stable_baselines3 import SAC
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;OUT=BASE/'outputs_v2d'/'attempt_01';sys.path.insert(0,str(ROOT/'src'))
sp=importlib.util.spec_from_file_location('v2a',BASE/'run_v2a_demo.py');v=importlib.util.module_from_spec(sp);sys.modules['v2a']=v;sp.loader.exec_module(v)
P={'pd_zero':('pd_gravity_residual',None,'original'),'pd_sac':('pd_gravity_residual',ROOT/'policies/pd_gravity_posthoc/seed_201/best_model.zip','original'),'pd_phase500':('pd_gravity_residual',ROOT/'policies/pd_gravity_posthoc/seed_201/best_model.zip','clamp500'),'rbf_zero':('rbf_residual',None,'original'),'rbf_sac':('rbf_residual',ROOT/'policies/p4b/rbf_residual/seed_201/best_model.zip','original'),'rbf_phase500':('rbf_residual',ROOT/'policies/p4b/rbf_residual/seed_201/best_model.zip','clamp500')}
def sha(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def run(name,mode,path,phase,ref):
 e=v.make_env(mode,ref);pol=None if path is None else SAC.load(path,device='cpu');obs=e._obs(*e._reference());rows=[];initial={'q':e.d.qpos[e.s.qadr].tolist(),'dq':e.d.qvel[e.s.dof].tolist(),'physics_time_s':e.d.time}
 for k in range(len(ref.q)-1):
  inp=obs.copy();inp[35]=min(k/500,1) if phase=='clamp500' else inp[35]
  act=np.zeros(7) if pol is None else pol.predict(inp,deterministic=True)[0];obs,reward,term,trunc,info=e.step(act);qr,_,_=e._reference();q=e.d.qpos[e.s.qadr].copy();ee=v.fk(e,q);er=v.fk(e,qr)
  rows.append({'step':k+1,'time_s':(k+1)*.004,'policy_phase_input':float(inp[35]),'environment_phase':float(obs[35]),'q':q.tolist(),'qref':qr.tolist(),'joint_rms_rad':float(np.sqrt(np.mean((q-qr)**2))),'ee_error_m':float(np.linalg.norm(ee-er)),'action':np.asarray(act).tolist(),'residual_norm_nm':float(np.linalg.norm(np.asarray(act)*e.residual_scale)),'residual_abs_nm':(np.asarray(act)*e.residual_scale).tolist(),'action_saturation_fraction':float(np.mean(np.abs(act)>=.999)),'raw_torque_nm':np.asarray(info['raw_torque_nm']).tolist(),'applied_torque_nm':np.asarray(info['safe_torque_nm']).tolist(),'projection_event':bool(info['projected']),'projection_norm_nm':float(info['qp'].get('projection_norm_nm',0)),'termination_reason':info['termination_reason']})
  if term or trunc:break
 e.close();return rows,initial
def stats(x,a,b):
 z=[r for r in x if a<r['time_s']<=b];
 if not z:return {}
 return {'n':len(z),'mean_joint_rms_rad':float(np.mean([r['joint_rms_rad'] for r in z])),'max_joint_rms_rad':float(np.max([r['joint_rms_rad'] for r in z])),'mean_ee_error_cm':float(100*np.mean([r['ee_error_m'] for r in z])),'max_ee_error_cm':float(100*np.max([r['ee_error_m'] for r in z])),'mean_residual_norm_nm':float(np.mean([r['residual_norm_nm'] for r in z])),'projection_fraction':float(np.mean([r['projection_event'] for r in z])),'mean_projection_norm_nm':float(np.mean([r['projection_norm_nm'] for r in z]))}
def main():
 if OUT.exists():raise SystemExit('append only')
 OUT.mkdir(parents=True);probe=v.make_env('pd_gravity_residual',v.Reference('p',np.zeros((2,7))));home=probe.home.copy();probe.close();ref=v.sweep(home);all={};initials={}
 for n,(m,p,ph) in P.items():all[n],initials[n]=run(n,m,p,ph,ref);(OUT/f'{n}_records.json').write_text(json.dumps(all[n],indent=2)+'\n')
 if len({json.dumps(x,sort_keys=True) for x in initials.values()})!=1:raise RuntimeError('initial mismatch')
 rows=[]
 for n,x in all.items():
  for label,a,b in [('0_2',0,2),('2_6',2,6),('0_9_5',0,9.5)]: rows.append({'controller':n,'window':label,'completion':len(x)==2375,'failure_time_s':x[-1]['time_s'] if x[-1]['termination_reason'] else '', 'reason':x[-1]['termination_reason'] or 'horizon_complete',**stats(x,a,b)})
 with open(OUT/'metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True)
 for n in ['pd_zero','pd_sac','pd_phase500','rbf_zero','rbf_sac','rbf_phase500']:
  x=all[n];t=[r['time_s'] for r in x];ax[0].plot(t,[100*r['ee_error_m'] for r in x],label=n);ax[1].plot(t,[r['residual_norm_nm'] for r in x],label=n);ax[2].plot(t,[r['projection_norm_nm'] for r in x],label=n)
 ax[0].set_ylabel('EE error (cm)');ax[1].set_ylabel('SAC residual (Nm)');ax[2].set_ylabel('QP correction (Nm)');ax[2].set_xlabel('physical control time (s)');ax[0].legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(OUT/'paired_time_series.png',dpi=160);plt.close(fig)
 man={'classification':'DEMO_ONLY / EXPLORATORY_DIAGNOSIS','formal_protocol_modified':False,'observation_phase_index':35,'phase_counterfactual':'policy input only: min(k/500,1); environment/plant/safety/policy weights unchanged','training_tracking_controls':500,'demo_tracking_controls':2375,'initial_state':initials['pd_zero'],'policies':{n:(str(p.relative_to(ROOT)) if p else None) for n,(_,p,_) in P.items()},'policy_hashes':{n:sha(p) for n,(_,p,_) in P.items() if p},'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}};(OUT/'MANIFEST.json').write_text(json.dumps(man,indent=2)+'\n')
if __name__=='__main__':main()

"""Frozen P5 zero-shot mismatch campaign; no policy training or selection."""
import csv, hashlib, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from p3_fer_qp_v2_env import PandaResidualFERQPV2Env, protocol, scenario

ROOT=Path(__file__).resolve().parents[1]
ADDENDUM=ROOT/'r3_mujoco/P5_ZERO_SHOT_FROZEN_ADDENDUM_V1.json'
P4B=ROOT/'r3_mujoco/p4b_full_attempt02'
OUT=Path(os.environ.get('P5_OUTPUT_DIR',ROOT/'r3_mujoco/p5_zero_shot_attempt01'))
MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual'); POLICY_SEEDS=(201,202,203,204,205)

def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def addendum():return json.loads(ADDENDUM.read_text())
def payload_xml(condition):
 """Create a derived XML with a non-colliding fixed inertial sphere under hand."""
 mass=condition.get('payload_mass_kg',0.0)
 if not mass:return ROOT/'r3_mujoco/panda_direct_torque_v2.xml'
 radius=.03;inertia=2*mass*radius*radius/5
 base=(ROOT/'r3_mujoco/panda_direct_torque_v2.xml').read_text()
 marker='<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
 inject=marker+f'\n                      <body name="p5_payload_{condition["id"]}" pos="0 0 0.10"><inertial pos="0 0 0" mass="{mass:.12g}" diaginertia="{inertia:.12g} {inertia:.12g} {inertia:.12g}"/></body>'
 if base.count(marker)!=1:raise RuntimeError('payload insertion marker not unique')
 # Keep this beside the source MJCF: asset paths in the Menagerie-derived XML
 # are relative to r3_mujoco, not an ephemeral results directory.
 out=ROOT/'r3_mujoco'/f'p5_derived_{condition["id"]}.xml'
 tmp=out.with_name(out.name+f'.{os.getpid()}.tmp');tmp.write_text(base.replace(marker,inject));os.replace(tmp,out);return out
def mutator(condition):
 def apply(m):
  for i in range(1,8):
   bid=m.body(f'link{i}').id
   if condition.get('mass_inertia_scale') is not None:
    x=float(condition['mass_inertia_scale']);m.body_mass[bid]*=x;m.body_inertia[bid]*=x
   elif condition.get('inertia_scale') is not None:m.body_inertia[bid]*=float(condition['inertia_scale'])
   did=m.joint(f'joint{i}').dofadr[0]
   if condition.get('damping_scale') is not None:m.dof_damping[did]*=float(condition['damping_scale'])
   if condition.get('frictionloss_nm') is not None:m.dof_frictionloss[did]=float(condition['frictionloss_nm'][i-1])
 return apply
def model_snapshot(condition):
 xml=payload_xml(condition);m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);mutator(condition)(m);mujoco.mj_setConst(m,d)
 links=[]
 for i in range(1,8):
  bid=m.body(f'link{i}').id;did=m.joint(f'joint{i}').dofadr[0];links.append({'link':i,'mass':float(m.body_mass[bid]),'inertia':m.body_inertia[bid].tolist(),'damping':float(m.dof_damping[did]),'frictionloss':float(m.dof_frictionloss[did])})
 # Positive diagonal inertias are necessary but not sufficient; MuJoCo compilation and mj_setConst above additionally validate model structure.
 if any(min(x['inertia'])<=0 or x['mass']<=0 for x in links):raise ValueError('nonphysical mass/inertia')
 return {'id':condition['id'],'derived_xml':str(xml),'derived_xml_sha256':sha(xml),'links':links,'payload_mass_kg':condition.get('payload_mass_kg',0.0)}
def make_env(mode,condition,split,seed):return PandaResidualFERQPV2Env(mode,split,seed,model_xml_path=payload_xml(condition),model_mutator=mutator(condition))
def preflight():
 a=addendum();checks=[]
 # Train-side only: nominal base env equals explicit nominal mismatch env through same actions.
 seed=10000;actions=np.zeros((8,7),dtype=np.float32);base=PandaResidualFERQPV2Env('pd_residual','train',seed);nom=make_env('pd_residual',a['conditions'][0],'train',seed);o1,_=base.reset(seed=seed);o2,_=nom.reset(seed=seed);delta=float(np.max(abs(o1-o2)))
 for x in actions:
  o1,r1,t1,tr1,i1=base.step(x);o2,r2,t2,tr2,i2=nom.step(x)
  delta=max(delta,float(np.max(abs(o1-o2))),abs(r1-r2),float(np.max(abs(i1['safe_torque_nm']-i2['safe_torque_nm']))))
  if t1 or tr1 or t2 or tr2:break
 if delta>1e-10:raise RuntimeError(f'nominal equivalence failure {delta}')
 snapshots=[model_snapshot(c) for c in a['conditions']]
 # Train-side, deterministic smoke only: confirms every mutated forward plant
 # compiles, resets through the unchanged startup safety path, and accepts a
 # finite command.  It never constructs a Final Holdout environment.
 plant_smoke=[]
 for c in a['conditions']:
  env=make_env('pd_residual',c,'train',10000);obs,_=env.reset(seed=10000)
  obs,reward,term,trunc,info=env.step(np.zeros(7,dtype=np.float32))
  if not np.isfinite(np.r_[obs,reward,info['safe_torque_nm']]).all():raise FloatingPointError('plant smoke nonfinite '+c['id'])
  plant_smoke.append({'condition':c['id'],'terminated':bool(term),'truncated':bool(trunc),'termination_reason':info['termination_reason'],'finite':True})
 # Check 30 final seeds before final evaluation, without instantiating plant/controller.
 scen=[scenario(s,'final') for s in a['evaluation']['final_holdout_seeds']]; keys=[(x['task'],x['amplitude_multiplier'],x['frequency_hz'],x['phase_offset_rad'],x['single_joint_index']) for x in scen]
 atomic(OUT/'PREFLIGHT.json',{'state':'PASS','nominal_train_side_max_delta':delta,'conditions':snapshots,'train_side_plant_smoke':plant_smoke,'final_scenarios':scen,'unique_reference_trajectory_count':len(set(keys)),'total_final_seeds':len(scen),'final_holdout_evaluated':False})
def episode(model,mode,policy_seed,condition,scene_seed):
 env=make_env(mode,condition,'final',scene_seed);obs,_=env.reset(seed=scene_seed);qse=dqse=tau2=proj2=0.;ret=0.;n=proj=0;reason=''
 while True:
  action,_=model.predict(obs,deterministic=True);obs,reward,term,trunc,info=env.step(action);q=np.asarray(env.d.qpos[env.s.qadr]);dq=np.asarray(env.d.qvel[env.s.dof]);qr,vr,_=env._reference();raw=np.asarray(info['raw_torque_nm']);safe=np.asarray(info['safe_torque_nm'])
  if not np.isfinite(np.r_[q,dq,qr,vr,raw,safe]).all():raise FloatingPointError('nonfinite')
  qse+=float(np.sum((q-qr)**2));dqse+=float(np.sum((dq-vr)**2));tau2+=float(np.sum(safe**2));proj2+=float(np.sum((raw-safe)**2));ret+=float(reward);n+=1;proj+=int(info['projected'])
  if term or trunc:reason=str(info.get('termination_reason',''));break
 return {'mode':mode,'policy_seed':policy_seed,'condition':condition['id'],'scenario_seed':scene_seed,'task':info['scenario']['task'],'steps':n,'return':ret,'terminated':int(bool(reason)),'termination_reason':reason,'projection_steps':proj,'q_sse':qse,'dq_sse':dqse,'safe_tau_sse':tau2,'raw_minus_safe_sse':proj2,'q_rmse_prefix_rad':float(np.sqrt(qse/(n*7))),'dq_rmse_prefix_rad_s':float(np.sqrt(dqse/(n*7)))}
def lane(mode):
 os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
 import torch;torch.set_num_threads(1)
 a=addendum();out=OUT/mode;out.mkdir(parents=True,exist_ok=True);rows=[];start=time.perf_counter()
 for seed in POLICY_SEEDS:
  modelpath=P4B/mode/f'seed_{seed:03d}'/'best_model.zip';model=SAC.load(modelpath,device='cpu');mh=sha(modelpath)
  for cond in a['conditions']:
   for scene in a['evaluation']['final_holdout_seeds']:
    row=episode(model,mode,seed,cond,int(scene));row['best_model_sha256']=mh;rows.append(row)
  atomic(out/f'seed_{seed:03d}_status.json',{'state':'COMPLETE','episodes':len(a['conditions'])*len(a['evaluation']['final_holdout_seeds']),'policy_sha256':mh})
 with (out/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return {'mode':mode,'state':'COMPLETE','episodes':len(rows),'wall_s':time.perf_counter()-start}
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir(parents=True);preflight();a=addendum();atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','conditions':len(a['conditions']),'episodes_expected':9750,'final_holdout_accessed':True,'parallelism':5,'started_unix_s':time.time()});t=time.perf_counter();results=[]
 with ProcessPoolExecutor(max_workers=5) as ex:
  fs={ex.submit(lane,m):m for m in MODES}
  for f in as_completed(fs):
   m=fs[f]
   try:results.append(f.result())
   except Exception as e:results.append({'mode':m,'state':'FAILED','error_type':type(e).__name__,'error':str(e)})
   atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','completed_lanes':results,'conditions':len(a['conditions']),'episodes_expected':9750,'final_holdout_accessed':True})
 state='COMPLETE' if len(results)==5 and all(x['state']=='COMPLETE' for x in results) else 'FAILED';atomic(OUT/'CAMPAIGN_STATUS.json',{'state':state,'results':results,'wall_s':time.perf_counter()-t,'conditions':len(a['conditions']),'episodes_expected':9750,'final_holdout_accessed':True})
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

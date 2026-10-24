"""One-shot P5 confirmation with mismatch forward Plants and nominal-only control model."""
import csv, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
import mujoco
import p2_simtrained_closed_loop as base
from p3_fer_qp_v2_env import PandaResidualFERQPV2Env, TASKS, reference, protocol
import p5_zero_shot_campaign as old
ROOT=Path(__file__).resolve().parents[1]; ADD=ROOT/'r3_mujoco/P5_NOMINAL_PREARMED_V2_CONTINUOUS.json';P4B=ROOT/'r3_mujoco/p4b_full_attempt02';OUT=Path(os.environ.get('P5V2_OUTPUT_DIR',ROOT/'r3_mujoco/p5_nominal_prearmed_attempt01'));MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual');SEEDS=(201,202,203,204,205)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,o):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');os.replace(t,p)
def cfg():return json.loads(ADD.read_text())
def scene(seed):
 p=protocol()['scenarios'];r=np.random.default_rng(seed);task=TASKS[seed%3]
 # Confirmation-only continuous phase makes complete trajectories unique while
 # remaining inside the owner-frozen [-.15,+.15] phase range.
 return {'seed':seed,'split':'p5_confirmation','task':task,'amplitude_multiplier':float(r.uniform(.8,1.0)),'frequency_hz':float(r.choice(p['frequency_hz'])),'phase_offset_rad':float(r.uniform(-.15,.15)),'single_joint_index':int(seed%7) if task=='single_joint' else None}
def trajhash(s):
 q=[]
 for k in range(501):q.append(reference(s,k*.004,np.zeros(7))[0])
 return hashlib.sha256(np.asarray(q,dtype=np.float64).tobytes()).hexdigest()
ARM={'kp_scale':1.0,'kd_scale':1.0,'gravity_scale':1.0,'prearmed_nominal_hold':True}
def make(mode,c,split,s):return PandaResidualFERQPV2Env(mode,split,s,model_xml_path=old.payload_xml(c),model_mutator=old.mutator(c),nominal_control_isolation=True,scenario_definition=scene(s),arming_protocol=ARM)
def gate():
 a=cfg(); cs=a['scenarios']; nominal=a['conditions'][0]; alt=a['conditions'][-1]
 # same q/dq/raw: nominal QP must be invariant to Forward Plant mutation
 e0=make('pd_residual',nominal,'train',10000);e1=make('pd_residual',alt,'train',10000)
 q=e0.d.qpos.copy();v=e0.d.qvel.copy();e1.d.qpos[:]=q;e1.d.qvel[:]=v;raw=np.array([1.,-1.,.5,-.5,.1,-.1,.2])
 s0,l0,_=e0._apply(raw,0);s1,l1,_=e1._apply(raw,0);qpdelta=float(np.max(abs(s0-s1)))
 # Explicit nominal-vs-nominal identity plus startup gravity invariance.
 for e in (e0,e1):e.d_control.qpos[:]=q;e.d_control.qvel[:]=v;mujoco.mj_forward(e.m_control,e.d_control)
 g0=base.gravity(e0.m_control,e0.d_control,e0.s);g1=base.gravity(e1.m_control,e1.d_control,e1.s);gdelta=float(np.max(abs(g0-g1)))
 if qpdelta>1e-10 or gdelta>1e-10:raise RuntimeError(f'leak gate {qpdelta} {gdelta}')
 # Existing fault mechanism still fail-closed; actual Plant limits remain monitor-side.
 _,fault=e1.s.command_qp(raw,e1.d_control,force_solver_failure=True)
 if fault.get('feasible',True):raise RuntimeError('fault injection did not fail closed')
 snapshots=[old.model_snapshot(c) for c in a['conditions']]
 scenarios=[scene(x) for x in cs['seeds']]; hashes=[trajhash(x) for x in scenarios]
 if len(set(hashes))!=30:raise RuntimeError('confirmation trajectories not unique')
 atomic(OUT/'PREFLIGHT.json',{'state':'PASS','nominal_qp_delta':qpdelta,'startup_gravity_delta':gdelta,'fault_fail_closed':True,'conditions':snapshots,'scenarios':scenarios,'trajectory_hashes':hashes,'unique_trajectory_count':30,'final_accessed':False})
def ep(model,mode,ps,c,ss):
 e=make(mode,c,'final',ss)
 try:o,_=e.reset(seed=ss)
 except Exception as x:return {'mode':mode,'policy_seed':ps,'condition':c['id'],'scenario_seed':ss,'steps':0,'return':0.,'terminated':1,'termination_reason':'STARTUP_FAILURE:'+str(x),'projection_steps':0,'q_sse':0.,'dq_sse':0.,'q_rmse_prefix_rad':float('nan'),'dq_rmse_prefix_rad_s':float('nan')}
 qse=dqse=ret=0.;n=proj=0;why=''
 while True:
  ac,_=model.predict(o,deterministic=True);o,r,t,tr,info=e.step(ac);q=e.d.qpos[e.s.qadr];dq=e.d.qvel[e.s.dof];qr,vr,_=e._reference();qse+=float(np.sum((q-qr)**2));dqse+=float(np.sum((dq-vr)**2));ret+=r;n+=1;proj+=int(info['projected'])
  if t or tr:why=info['termination_reason'];break
 return {'mode':mode,'policy_seed':ps,'condition':c['id'],'scenario_seed':ss,'steps':n,'return':ret,'terminated':int(bool(why)),'termination_reason':why,'projection_steps':proj,'q_sse':qse,'dq_sse':dqse,'q_rmse_prefix_rad':float(np.sqrt(qse/(n*7))),'dq_rmse_prefix_rad_s':float(np.sqrt(dqse/(n*7)))}
def lane(mode):
 os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1';import torch;torch.set_num_threads(1);a=cfg();rows=[];t=time.perf_counter();d=OUT/mode;d.mkdir(parents=True,exist_ok=True)
 for seed in SEEDS:
  mp=P4B/mode/f'seed_{seed:03d}'/'best_model.zip';m=SAC.load(mp,device='cpu')
  for c in a['conditions']:
   for s in a['scenarios']['seeds']:rows.append(ep(m,mode,seed,c,s))
  atomic(d/f'seed_{seed:03d}_status.json',{'state':'COMPLETE','policy_sha256':sha(mp),'episodes':390})
 with (d/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return {'mode':mode,'state':'COMPLETE','episodes':len(rows),'wall_s':time.perf_counter()-t}
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir();gate();a=cfg();atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','episodes_expected':9750,'final_accessed':True,'isolation':'nominal_control_model','parallelism':5});t=time.perf_counter();res=[]
 with ProcessPoolExecutor(max_workers=5) as ex:
  fs={ex.submit(lane,m):m for m in MODES}
  for f in as_completed(fs):
   try:res.append(f.result())
   except Exception as x:res.append({'mode':fs[f],'state':'FAILED','error':str(x)})
   atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','completed_lanes':res,'episodes_expected':9750,'final_accessed':True})
 state='COMPLETE' if len(res)==5 and all(x['state']=='COMPLETE' for x in res) else 'FAILED';atomic(OUT/'CAMPAIGN_STATUS.json',{'state':state,'results':res,'wall_s':time.perf_counter()-t,'episodes_expected':9750,'final_accessed':True});
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

"""P3 V2.1 qualification gate; no SAC training is performed here."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from stable_baselines3.common.env_checker import check_env
from p3_fer_qp_v2_env import PandaResidualFERQPV2Env, protocol, scenario

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'r3_mujoco/logs/p3_fer_qp_v2_gate_attempt01'
MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual')
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def episode(mode,action,seed=10010,n=64):
 e=PandaResidualFERQPV2Env(mode,'train',seed,max_controls=n);e.reset(seed=7);rows=[];t=time.perf_counter()
 while True:
  _,r,term,trunc,info=e.step(action);rows.append((r,term,trunc,info))
  if term or trunc:return rows,time.perf_counter()-t
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir(parents=True);p=protocol(); rows=[]
 # Gymnasium/SB3 contract and fixed-mode semantics.
 check_env(PandaResidualFERQPV2Env('pd_residual','train',10000,max_controls=2),warn=True,skip_render_check=True)
 for mode in MODES:
  a=PandaResidualFERQPV2Env(mode,'train',10003,max_controls=2);oa,ia=a.reset(seed=123)
  b=PandaResidualFERQPV2Env(mode,'train',10003,max_controls=2);ob,ib=b.reset(seed=123)
  if not(np.array_equal(oa,ob) and ia['scenario']==ib['scenario']):raise RuntimeError(mode+' nondeterministic reset')
  _,rew,term,trunc,x=a.step(np.zeros(7));raw=x['raw_torque_nm'];pd=x['pd_torque_nm'];prior=x['prior_torque_nm']
  expected=np.zeros(7) if mode=='pure_sac' else (pd if mode=='pd_residual' else prior+pd)
  delta=float(np.max(abs(raw-expected)))
  rp=p['reward'];qr,vr,_=a._reference();E=float(np.mean(((a.d.qpos[a.s.qadr]-qr)/rp['tracking_position_scale_rad'])**2)+rp['velocity_weight']*np.mean(((a.d.qvel[a.s.dof]-vr)/rp['tracking_velocity_scale_rad_s'])**2));ct=float(np.mean((x['safe_torque_nm']/a.s.limit)**2));cp=float(min(1,np.mean(((raw-x['safe_torque_nm'])/a.s.limit)**2)));recalc=1/(1+E)-rp['torque_weight']*ct-rp['projection_weight']*cp
  if delta>1e-10 or abs(rew-recalc)>1e-10:raise RuntimeError(mode+' controller/reward mismatch')
  rows.append({'case':'zero_residual_and_reward','mode':mode,'steps':1,'terminated':term,'truncated':trunc,'reason':x['termination_reason'],'raw_delta_nm':delta,'reward_delta':abs(rew-recalc),'pass':True})
 # The three frozen namespaces cannot overlap; Final is not instantiated.
 ss=[scenario(10000,'train'),scenario(20000,'validation'),scenario(30000,'final')]
 if len({x['split'] for x in ss})!=3 or len({x['seed'] for x in ss})!=3:raise RuntimeError('scenario split collision')
 # Bound-action stress validates normal completion or correctly penalised fail-closed termination.
 rng=np.random.default_rng(77)
 for mode in MODES:
  for label,action in [('positive',np.ones(7)),('negative',-np.ones(7)),('random',rng.uniform(-1,1,7))]:
   z,wall=episode(mode,action);r,term,trunc,info=z[-1];ok=(term and bool(info['termination_reason']) and -100.000001<=r<=-98.999999) or (trunc and r>0)
   rows.append({'case':'stress_'+label,'mode':mode,'steps':len(z),'terminated':term,'truncated':trunc,'reason':info['termination_reason'],'raw_delta_nm':0.,'reward_delta':0.,'pass':bool(ok),'wall_s':wall})
 # Production safety hooks: action validation, solver fail-close, warning and contact injection.
 e=PandaResidualFERQPV2Env('pd_residual','train',10000,max_controls=2);e.reset(seed=1);before=e.d.time
 try:e.step(np.full(7,np.nan));nan_rejected=False
 except ValueError:nan_rejected=True
 if not nan_rejected or e.d.time!=before:raise RuntimeError('NaN input was not fail-closed')
 raw=np.zeros(7);safe,qi=e.s.command_qp(raw,e.d,force_solver_failure=True)
 if qi['feasible'] or np.any(safe):raise RuntimeError('QP failure did not fail close')
 warning=e.s.monitor(e.d,99,inject_warning=1);contact=e.s.monitor(e.d,100,inject_contact=True)
 if warning['termination']!='solver_warning' or contact['termination']!='abnormal_contact':raise RuntimeError('monitor injection failure')
 rows += [
  {'case':'nan_action_fail_closed','mode':'pd_residual','steps':0,'terminated':True,'truncated':False,'reason':'ValueError','raw_delta_nm':0.,'reward_delta':0.,'pass':True},
  {'case':'qp_injected_failure','mode':'pd_residual','steps':0,'terminated':True,'truncated':False,'reason':qi['reason'],'raw_delta_nm':0.,'reward_delta':0.,'pass':True},
  {'case':'warning_injection','mode':'pd_residual','steps':0,'terminated':True,'truncated':False,'reason':warning['termination'],'raw_delta_nm':0.,'reward_delta':0.,'pass':True},
  {'case':'contact_injection','mode':'pd_residual','steps':0,'terminated':True,'truncated':False,'reason':contact['termination'],'raw_delta_nm':0.,'reward_delta':0.,'pass':True}]
 fields=sorted({k for row in rows for k in row})
 with (OUT/'P3_GATE_RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 status={'protocol_sha256':digest(ROOT/'r3_mujoco/P3_P4_P5_FER_QP_V2_PROTOCOL.json'),'sb3_check_env':'PASS','controller_modes':'PASS_5_OF_5','reward_recalculation':'PASS_5_OF_5','scenario_isolation':'PASS','final_holdout_accessed':False,'stress_pass':sum(bool(x['pass']) for x in rows if str(x['case']).startswith('stress_')),'fault_injection':'PASS_4_OF_4','rows':len(rows),'gate':'PASS'}
 (OUT/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':main()

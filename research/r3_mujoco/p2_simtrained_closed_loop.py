"""P2 closed-loop qualification for Owner-frozen Sim-trained priors."""
import csv, hashlib, json, sys, time
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'r3_mujoco'));sys.path.insert(0,str(ROOT/'r3_mujoco/tests'))
from safety import PandaSafety
from startup_policy_v2 import safehold_v2
import p0_handoff as handoff
from p2_simtrained_adapters import predict_npz
P=ROOT/'r3_mujoco/P2_SIMTRAINED_QUALIFICATION_V1.json';O=ROOT/'r3_mujoco/logs/p2_simtrained_closed_loop_v1'
KP=handoff.CP;KD=handoff.CD;DT=.004;SUB=2
ART={'near_ols_3000':ROOT/'r3_mujoco/p2_offline/artifacts/near_ols_3000.npz','rbf_128_w6_a01_3000':ROOT/'r3_mujoco/p2_offline/artifacts/rbf_128_w6_a01_3000.npz','mlp64x2_3000_41':ROOT/'r3_mujoco/p2_offline/artifacts/mlp64x2_3000_41.npz'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ref(kind,t,home):
 amp=np.array([.16,.10,.14,.10,.08,.10,.06])
 if kind=='single_joint':amp=np.array([.20,0,0,0,0,0,0])
 if kind=='point_to_point':
  u=min(t,1.);return home+amp*(.5-.5*np.cos(np.pi*u)),amp*(.5*np.pi*np.sin(np.pi*u) if t<1 else 0),amp*(.5*np.pi**2*np.cos(np.pi*u) if t<1 else 0)
 w=2*np.pi*.35;phase=np.arange(7)*.23;tr=.4;r=min(max(t/tr,0),1);s=10*r**3-15*r**4+6*r**5 if r<1 else 1;sd=(30*r*r-60*r**3+30*r**4)/tr if r<1 else 0;sdd=(60*r-180*r*r+120*r**3)/(tr*tr) if r<1 else 0
 b=amp*np.sin(w*t+phase);bd=amp*w*np.cos(w*t+phase);bdd=-amp*w*w*np.sin(w*t+phase)
 return home+s*b,sd*b+s*bd,sdd*b+2*sd*bd+s*bdd
def gravity(m,d,s):
 v=d.qvel.copy();d.qvel[:]=0;mujoco.mj_forward(m,d);o=d.qfrc_bias[s.dof]-d.qfrc_passive[s.dof];d.qvel[:]=v;mujoco.mj_forward(m,d);return o
def run(controller,task,cfg):
 m=mujoco.MjModel.from_xml_path(str(ROOT/'r3_mujoco/panda_direct_torque_v2.xml'));m.opt.timestep=.002
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);s=PandaSafety(m);s.reset();s.monitor_reset(d);home=d.qpos[s.qadr].copy();rows=[];term='';start=time.perf_counter()
 for k in range(cfg['startup_controls']+cfg['tracking_controls']):
  phase='startup' if k<cfg['startup_controls'] else 'tracking'
  if phase=='startup':
   hold=safehold_v2(m,d,s,home);ok,why=handoff.lp_result_safe(hold)
   if not ok:term=why;break
   raw=hold.x[:7];qr=home.copy();vr=ar=np.zeros(7)
  else:
   qr,vr,ar=ref(task,(k-cfg['startup_controls'])*DT,home);pd=KP*(qr-d.qpos[s.qadr])+KD*(vr-d.qvel[s.dof])
   if controller=='pd_gravity':raw=pd+gravity(m,d,s)
   elif controller=='computed_torque':
    mujoco.mj_forward(m,d);mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM);raw=mass[np.ix_(s.dof,s.dof)]@(ar+KD*(vr-d.qvel[s.dof])+KP*(qr-d.qpos[s.qadr]))+d.qfrc_bias[s.dof]-d.qfrc_passive[s.dof]
   else:raw=predict_npz(ART[controller],np.r_[d.qpos[s.qadr],d.qvel[s.dof],ar][None])[0]+pd
  safe,lim=s.command_acceleration_aware(raw,d,cfg['prediction_acceleration_margin'])
  if not lim['feasible']:term='acceleration_projection_infeasible';break
  d.ctrl[:7]=safe
  for sub in range(SUB):
   mujoco.mj_step(m,d);ev=s.monitor(d,2*k+sub);term=ev['termination'] or ''
   rows.append((d.time,phase,d.qpos[s.qadr].copy(),d.qvel[s.dof].copy(),d.qacc[s.dof].copy(),qr.copy(),safe.copy(),d.qfrc_actuator[s.dof].copy(),term))
   if term:break
  if term:break
 t=np.array([r[0] for r in rows]);ph=np.array([r[1] for r in rows],dtype='U16');q=np.stack([r[2] for r in rows]);dq=np.stack([r[3] for r in rows]);ddq=np.stack([r[4] for r in rows]);qr=np.stack([r[5] for r in rows]);safe=np.stack([r[6] for r in rows]);app=np.stack([r[7] for r in rows]);tr=ph=='tracking';err=q[tr]-qr[tr]
 np.savez_compressed(O/f'{controller}_{task}.npz',time=t,phase=ph,q=q,qvel=dq,qacc=ddq,qref=qr,safe=safe,applied=app,termination=np.asarray([r[8] for r in rows],dtype='U32'))
 ratios=[np.max(abs(ddq)/s.acc),np.max(abs(app)/s.limit),np.max(abs(dq)/s.vel)];slew=np.max(abs(np.diff(safe[::2],axis=0))/(DT*s.slew)) if len(safe)>2 else np.inf
 return {'controller':controller,'task':task,'physics_steps':len(rows),'termination':term,'tracking_rmse_rad':float(np.sqrt(np.mean(err*err))),'tracking_mae_rad':float(np.mean(abs(err))),'torque_rms_nm':float(np.sqrt(np.mean(safe*safe))),'acceleration_ratio':float(ratios[0]),'torque_ratio':float(ratios[1]),'velocity_ratio':float(ratios[2]),'slew_ratio':float(slew),'wall_s':time.perf_counter()-start,'pass':bool(not term and len(rows)==2000 and np.isfinite(np.r_[q,dq,ddq,safe,app]).all() and max(*ratios,slew)<=1+1e-12),'artifact':f'{controller}_{task}.npz'}
def main():
 if O.exists():raise RuntimeError('append-only output exists')
 cfg=json.loads(P.read_text());O.mkdir(parents=True);rows=[run(c,t,cfg) for c in cfg['controllers'] for t in cfg['tasks']]
 with (O/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 status={'protocol_sha256':sha(P),'artifact_sha256':{k:sha(v) for k,v in ART.items()},'episodes':len(rows),'pass_count':sum(x['pass'] for x in rows),'aggregate_pass':all(x['pass'] for x in rows),'selection':'OWNER_FROZEN_mlp64x2_3000_41'}
 (O/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':main()

"""New P3 environment; historical P3 environment remains immutable."""
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco, numpy as np

from .paths import ASSETS, CONFIGS, ROOT
from .safety_fer_qp_v2 import PandaSafetyFERQPV2
from . import nominal as base
from .priors import load_resident

PROTOCOL=CONFIGS/'P3_P4_P5_FER_QP_V2_PROTOCOL.json'
TASKS=('point_to_point','single_joint','multi_joint')
def protocol(): return json.loads(PROTOCOL.read_text())
def scenario(seed,split):
 p=protocol()['scenarios'];lo,hi=p['train_seed_range']
 if split=='train' and not lo<=seed<=hi: raise ValueError('train seed out of frozen range')
 if split=='validation' and seed not in p['validation_seeds']: raise ValueError('validation seed not frozen')
 if split=='final' and seed not in p['final_holdout_seeds']: raise ValueError('final seed not frozen')
 r=np.random.default_rng(int(seed)); task=TASKS[int(seed)%3]
 return {'seed':int(seed),'split':split,'task':task,'amplitude_multiplier':float(r.choice(p['amplitude_multiplier'])),'frequency_hz':float(r.choice(p['frequency_hz'])),'phase_offset_rad':float(r.choice(p['phase_offset_rad'])),'single_joint_index':int(seed%7) if task=='single_joint' else None}
def reference(s,t,home):
 amp=np.array([.16,.10,.14,.10,.08,.10,.06])*s['amplitude_multiplier']
 if s['task']=='single_joint': amp=np.eye(7)[s['single_joint_index']]*(.20*s['amplitude_multiplier'])
 if s['task']=='point_to_point':
  u=min(t,1.); return home+amp*(.5-.5*np.cos(np.pi*u)),amp*(.5*np.pi*np.sin(np.pi*u) if t<1 else 0),amp*(.5*np.pi**2*np.cos(np.pi*u) if t<1 else 0)
 w=2*np.pi*s['frequency_hz'];ph=np.arange(7)*.23+s['phase_offset_rad'];r=min(max(t/.4,0),1);g=10*r**3-15*r**4+6*r**5 if r<1 else 1;gd=(30*r*r-60*r**3+30*r**4)/.4 if r<1 else 0;gdd=(60*r-180*r*r+120*r**3)/.16 if r<1 else 0;b=amp*np.sin(w*t+ph);bd=amp*w*np.cos(w*t+ph);bdd=-amp*w*w*np.sin(w*t+ph)
 return home+g*b,gd*b+g*bd,gdd*b+2*gd*bd+g*bdd
class PandaResidualFERQPV2Env(gym.Env):
 metadata={'render_modes':[]}
 def __init__(self,mode,split='train',scenario_seed=None,max_controls=None,model_xml_path=None,model_mutator=None,nominal_control_isolation=False,scenario_definition=None,arming_protocol=None):
  super().__init__();self.p=protocol()
  if mode not in ('pure_sac','pd_residual','pd_gravity_residual','mlp_residual','gp_residual','rbf_residual'): raise ValueError('invalid controller mode')
  if split not in ('train','validation','final'): raise ValueError('invalid split')
  self.mode,self.split=mode,split;self.m=mujoco.MjModel.from_xml_path(str(model_xml_path or ASSETS/'panda_direct_torque_v2.xml'));self.m.opt.timestep=self.p['timebase']['physics_dt_s']
  if model_mutator is not None: model_mutator(self.m)
  self.d=mujoco.MjData(self.m)
  if model_mutator is not None: mujoco.mj_setConst(self.m,self.d)
  self.nominal_control_isolation=bool(nominal_control_isolation)
  if self.nominal_control_isolation:
   # The control-side model is deliberately a separate, unperturbed nominal
   # model. Only q/qdot observations are copied into it; no Plant parameter,
   # Plant mass matrix or Plant bias term reaches QP or startup gravity.
   self.m_control=mujoco.MjModel.from_xml_path(str(ASSETS/'panda_direct_torque_v2.xml'));self.m_control.opt.timestep=self.p['timebase']['physics_dt_s'];self.d_control=mujoco.MjData(self.m_control)
   self.s=PandaSafetyFERQPV2(self.m_control);self.s_monitor=PandaSafetyFERQPV2(self.m)
  else:
   self.m_control,self.d_control=None,None;self.s=PandaSafetyFERQPV2(self.m);self.s_monitor=self.s
  self.prior=load_resident(ROOT/self.p['priors'][mode]['path']) if mode in self.p['priors'] else None
  self.residual_scale=np.asarray(self.p['action']['residual_scale_nm'],float);self.pure_scale=np.asarray(self.p['action']['pure_scale_nm'],float);self.max_controls=max_controls or self.p['timebase']['tracking_controls'];self.forced_seed=scenario_seed;self.scenario_definition=None if scenario_definition is None else dict(scenario_definition);self.arming_protocol=arming_protocol;self.k=0;self.home=None;self.scenario=None
  self.action_space=spaces.Box(-1.,1.,shape=(7,),dtype=np.float32);self.observation_space=spaces.Box(-np.inf,np.inf,shape=(39,),dtype=np.float32)
 def _pick_seed(self):
  if self.forced_seed is not None:return int(self.forced_seed)
  if self.split!='train':raise ValueError('validation/final requires explicit frozen scenario seed')
  lo,hi=self.p['scenarios']['train_seed_range'];return int(self.np_random.integers(lo,hi+1))
 def _reference(self):return reference(self.scenario,self.k*.004,self.home)
 def _obs(self,qr,vr,ar):
  q=self.d.qpos[self.s.qadr];dq=self.d.qvel[self.s.dof];rng=(self.m.jnt_range[self.s.jid,1]-self.m.jnt_range[self.s.jid,0])/2;one=np.zeros(3);one[TASKS.index(self.scenario['task'])]=1
  return np.r_[(q-qr)/.15,(dq-vr)/.5,(q-self.home)/rng,dq/self.s.vel,ar/self.s.acc,self.k/self.max_controls,one].astype(np.float32)
 def _apply(self,raw,control_index):
  if self.nominal_control_isolation:
   self.d_control.qpos[:]=self.d.qpos;self.d_control.qvel[:]=self.d.qvel;self.d_control.act[:]=self.d.act
   mujoco.mj_forward(self.m_control,self.d_control)
   safe,lim=self.s.command_qp(raw,self.d_control)
  else: safe,lim=self.s.command_qp(raw,self.d)
  if not lim['feasible']: return safe,lim,lim.get('reason','qp_infeasible')
  self.d.ctrl[:7]=safe
  for sub in range(2):
   mujoco.mj_step(self.m,self.d);ev=self.s_monitor.monitor(self.d,2*control_index+sub)
   if ev['termination']: return safe,lim,ev['termination']
  return safe,lim,''
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);self.scenario=dict(self.scenario_definition) if self.scenario_definition is not None else scenario(self._pick_seed(),self.split);mujoco.mj_resetDataKeyframe(self.m,self.d,0);self.s.reset();self.s_monitor.reset();self.s_monitor.monitor_reset(self.d);self.home=self.d.qpos[self.s.qadr].copy();self.k=0
  if self.arming_protocol and self.arming_protocol.get('prearmed_nominal_hold',False):
   self.d_control.qpos[:]=self.d.qpos;self.d_control.qvel[:]=self.d.qvel;mujoco.mj_forward(self.m_control,self.d_control);hold=base.gravity(self.m_control,self.d_control,self.s)
   self.s.prev=hold.copy();self.d.ctrl[:7]=hold
  for j in range(self.p['timebase']['startup_controls']):
   q=self.d.qpos[self.s.qadr];dq=self.d.qvel[self.s.dof]
   if self.nominal_control_isolation:
    self.d_control.qpos[:]=self.d.qpos;self.d_control.qvel[:]=self.d.qvel;mujoco.mj_forward(self.m_control,self.d_control);g=base.gravity(self.m_control,self.d_control,self.s)
   else:g=base.gravity(self.m,self.d,self.s)
   if self.arming_protocol:
    n=self.p['timebase']['startup_controls'];u=(j+1)/n;alpha=3*u*u-2*u*u*u
    # Pre-armed means the first control command is continuous with tau_prev.
    # It must never subsequently restart a gravity ramp from zero.
    alpha=1.0 if self.arming_protocol.get('prearmed_nominal_hold',False) else alpha
    raw=float(self.arming_protocol.get('gravity_scale',1.0))*alpha*g+float(self.arming_protocol['kp_scale'])*(base.KP*(self.home-q))+float(self.arming_protocol['kd_scale'])*(base.KD*(-dq))
   else: raw=base.KP*(self.home-q)+base.KD*(-dq)+g
   _,_,why=self._apply(raw,-500+j)
   if why: raise RuntimeError('startup safety termination: '+why)
  qr,vr,ar=self._reference();return self._obs(qr,vr,ar),{'scenario':self.scenario.copy(),'startup_controls':500,'mode':self.mode}
 def step(self,action):
  a=np.asarray(action,float)
  if a.shape!=(7,) or not np.isfinite(a).all() or np.any(abs(a)>1+1e-12):raise ValueError('invalid normalized action')
  qr,vr,ar=self._reference();q=self.d.qpos[self.s.qadr];dq=self.d.qvel[self.s.dof];pd=base.KP*(qr-q)+base.KD*(vr-dq)
  if self.mode=='pure_sac':raw=a*self.pure_scale
  elif self.mode=='pd_residual':raw=pd+a*self.residual_scale
  elif self.mode=='pd_gravity_residual':
   # Nominal analytical gravity exactly once; no learned prior is loaded.
   if self.nominal_control_isolation:
    self.d_control.qpos[:]=self.d.qpos;self.d_control.qvel[:]=self.d.qvel;mujoco.mj_forward(self.m_control,self.d_control);g=base.gravity(self.m_control,self.d_control,self.s)
   else:g=base.gravity(self.m,self.d,self.s)
   raw=pd+g+a*self.residual_scale
  else:raw=self.prior.predict(np.r_[q,dq,ar][None])[0]+pd+a*self.residual_scale
  safe,lim,reason=self._apply(raw,self.k);terminated=bool(reason);self.k+=1;qr2,vr2,ar2=self._reference();q2=self.d.qpos[self.s.qadr];dq2=self.d.qvel[self.s.dof];rp=self.p['reward'];E=float(np.mean(((q2-qr2)/rp['tracking_position_scale_rad'])**2)+rp['velocity_weight']*np.mean(((dq2-vr2)/rp['tracking_velocity_scale_rad_s'])**2));ct=float(np.mean((safe/self.s.limit)**2));cp=float(min(1,np.mean(((raw-safe)/self.s.limit)**2)));reward=1/(1+E)-rp['torque_weight']*ct-rp['projection_weight']*cp
  if terminated: reward+=rp['safety_termination'];truncated=False
  else: truncated=self.k>=self.max_controls;reward+=rp['completion_bonus'] if truncated else 0
  return self._obs(qr2,vr2,ar2),float(reward),terminated,truncated,{'scenario':self.scenario.copy(),'termination_reason':reason,'raw_torque_nm':raw,'safe_torque_nm':safe,'prior_torque_nm':np.zeros(7) if self.prior is None else raw-pd-a*self.residual_scale,'pd_torque_nm':pd,'E':E,'C_tau':ct,'C_projection':cp,'projected':bool(lim.get('projected',False)),'qp':lim}

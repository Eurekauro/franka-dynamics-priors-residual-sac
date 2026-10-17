"""Panda/FER V2 torque safety projection.  Historical safety.py remains untouched."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import Bounds, LinearConstraint, minimize

from .paths import CONFIGS
CFG=json.loads((CONFIGS/'P2_OFFICIAL_FER_QP_V2.json').read_text())
ADDENDUM=json.loads((CONFIGS/'P2_OFFICIAL_FER_QP_V2_1.json').read_text())

class PandaSafetyFERQPV2:
 def __init__(self,model,control_dt=None):
  self.m=model;self.dt=float(control_dt or CFG['timebase']['control_dt_s'])
  self.dof=np.array([model.jnt_dofadr[model.joint(f'joint{i}').id] for i in range(1,8)])
  self.jid=np.array([model.joint(f'joint{i}').id for i in range(1,8)])
  self.qadr=np.array([model.jnt_qposadr[j] for j in self.jid])
  self.limit=np.asarray(CFG['torque_nm'],float);self.slew=np.asarray(CFG['torque_slew_nm_s'],float)
  self.vel=np.asarray(CFG['velocity_rad_s'],float);self.acc=np.asarray(CFG['acceleration_rad_s2'],float)
  self.jerk=np.asarray(CFG['jerk_rad_s3'],float);self.qmin=np.asarray(CFG['position_rad']['min'],float);self.qmax=np.asarray(CFG['position_rad']['max'],float);self.acc_qp=self.acc-float(ADDENDUM['affine_acceleration_epsilon_rad_s2'])
  self.W=np.asarray(CFG['qp']['W_diagonal'],float);self.prev=np.zeros(7);self.prev_qacc=None;self.warning_counts=None
 def reset(self):self.prev[:]=0;self.prev_qacc=None
 def monitor_reset(self,data):self.warning_counts=data.warning.number.copy();self.prev_qacc=np.asarray(data.qacc[self.dof],float).copy()
 def _affine(self,data):
  mujoco.mj_forward(self.m,data);M=np.zeros((self.m.nv,self.m.nv));mujoco.mj_fullM(self.m,M,data.qM);M=M[np.ix_(self.dof,self.dof)]
  try:inv=np.linalg.solve(M,np.eye(7))
  except np.linalg.LinAlgError:raise RuntimeError('mass_matrix_singular')
  bias=data.qfrc_bias[self.dof]-data.qfrc_passive[self.dof]
  return inv,inv@bias
 def command_qp(self,raw,data,force_solver_failure=False):
  raw=np.asarray(raw,float)
  if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError('invalid_action')
  lower=np.maximum(-self.limit,self.prev-self.slew*self.dt);upper=np.minimum(self.limit,self.prev+self.slew*self.dt)
  horizon=2*self.m.opt.timestep
  alo=np.maximum(-self.acc_qp,(-self.vel-data.qvel[self.dof])/horizon);ahi=np.minimum(self.acc_qp,(self.vel-data.qvel[self.dof])/horizon)
  if np.any(alo>ahi):return np.zeros(7),{'feasible':False,'reason':'velocity_viability_empty','solver':'not_called','projected':True}
  try:G,offset=self._affine(data)
  except RuntimeError as e:return np.zeros(7),{'feasible':False,'reason':str(e),'solver':'not_called','projected':True}
  # alo <= G tau - offset <= ahi.
  A=np.r_[G,-G];b=np.r_[ahi+offset,-alo-offset]
  tol=float(CFG['qp']['constraint_tolerance'])
  feasible_raw=bool(np.all(raw>=lower-tol) and np.all(raw<=upper+tol) and np.all(A@raw<=b+tol))
  if feasible_raw:
   self.prev=raw.copy();return raw.copy(),{'feasible':True,'solver':'raw_feasible','projected':False,'objective':0.0,'constraint_max_violation':float(np.max(A@raw-b)),'projection_norm_nm':0.0}
  if force_solver_failure:return np.zeros(7),{'feasible':False,'reason':'injected_qp_failure','solver':'injected','projected':True}
  fun=lambda x:0.5*np.sum(self.W*(x-raw)**2)
  jac=lambda x:self.W*(x-raw)
  sol=minimize(fun,np.clip(raw,lower,upper),jac=jac,method='SLSQP',bounds=Bounds(lower,upper),constraints=[LinearConstraint(A,-np.inf*np.ones(14),b)],options={'ftol':float(CFG['qp']['ftol']),'maxiter':int(CFG['qp']['maxiter']),'disp':False})
  x=np.asarray(sol.x,float);violation=float(max(np.max(A@x-b),np.max(lower-x),np.max(x-upper))) if x.shape==(7,) and np.isfinite(x).all() else np.inf
  if not sol.success or violation>tol:
   return np.zeros(7),{'feasible':False,'reason':'qp_solver_failure' if not sol.success else 'qp_constraint_residual','solver':'SLSQP','solver_message':str(sol.message),'projected':True,'constraint_max_violation':violation}
  self.prev=x.copy();return x,{'feasible':True,'solver':'SLSQP','projected':True,'objective':float(fun(x)),'constraint_max_violation':violation,'projection_norm_nm':float(np.linalg.norm(x-raw)),'iterations':int(getattr(sol,'nit',-1))}
 def monitor(self,data,step,inject_warning=None,inject_contact=False):
  now=data.warning.number.copy();old=np.zeros_like(now) if self.warning_counts is None else self.warning_counts;delta=np.where(now>old)[0].tolist();self.warning_counts=now.copy()
  if inject_warning is not None:delta.append(int(inject_warning))
  pairs=[]
  for i in range(data.ncon):
   c=data.contact[i];pairs.append(tuple(sorted((self.m.geom(c.geom1).name or f'geom{c.geom1}',self.m.geom(c.geom2).name or f'geom{c.geom2}'))))
  allowed=getattr(self,'allowed_contact_pairs',set());cc='injected_abnormal' if inject_contact else ('allowed_contact' if pairs and all(x in allowed for x in pairs) else ('self_or_unknown_contact' if pairs else ('internal_constraint' if data.nefc>0 else 'none')))
  state=self.state(data);term='solver_warning' if delta else ('abnormal_contact' if inject_contact or cc=='self_or_unknown_contact' else state)
  a=np.asarray(data.qacc[self.dof],float);jerk=np.full(7,np.nan) if self.prev_qacc is None else (a-self.prev_qacc)/self.m.opt.timestep;self.prev_qacc=a.copy()
  return {'step':int(step),'warning_types':delta,'ncon':int(data.ncon),'nefc':int(data.nefc),'contact_pairs':pairs,'contact_class':cc,'jerk_rad_s3':jerk,'jerk_ratio':float(np.nanmax(abs(jerk)/self.jerk)) if np.isfinite(jerk).all() else None,'termination':term}
 def state(self,data):
  q=np.asarray(data.qpos[self.qadr]);v=np.asarray(data.qvel[self.dof]);a=np.asarray(data.qacc[self.dof])
  if not np.isfinite(np.r_[q,v,a]).all():return 'nonfinite'
  if np.any(q<self.qmin) or np.any(q>self.qmax):return 'position_limit'
  if np.any(abs(v)>self.vel):return 'velocity_limit'
  if np.any(abs(a)>self.acc):return 'acceleration_limit'
  return None

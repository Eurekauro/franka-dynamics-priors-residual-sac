#!/usr/bin/env python3
"""V2-B2 frozen-policy replay and synchronized DEMO_ONLY comparison renderer."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import imageio.v2 as imageio
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_propagation
from stable_baselines3 import SAC

ROOT=Path(__file__).resolve().parents[2]; BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs_v2b2'/'attempt_04'; DT=.004; FPS=30; W,H=1920,1080
sys.path.insert(0,str(ROOT/'src'))
MOD=importlib.util.spec_from_file_location('v2a',BASE/'run_v2a_demo.py'); V2A=importlib.util.module_from_spec(MOD); sys.modules['v2a']=V2A; MOD.loader.exec_module(V2A)
POLICIES={'rbf_residual':ROOT/'policies/p4b/rbf_residual/seed_201/best_model.zip','pd_gravity_residual':ROOT/'policies/pd_gravity_posthoc/seed_201/best_model.zip'}
LABELS={'rbf_residual':'RBF + PD + Residual SAC','pd_gravity_residual':'PD + Gravity + Residual SAC'}
COLORS={'rbf_residual':(168,93,207),'pd_gravity_residual':(40,177,191)}
AMBER=np.array([.96,.62,.13,.30]); CYAN=np.array([.05,.78,.82,1.]);

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def font(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def fk(env,q): return V2A.fk(env,np.asarray(q,float))

def replay(mode,ref):
 """One deterministic, unmodified environment replay from the common reset."""
 env=V2A.make_env(mode,ref); policy=SAC.load(POLICIES[mode],device='cpu'); obs=env._obs(*env._reference()); rows=[]; reason=''
 initial={'q':env.d.qpos[env.s.qadr].tolist(),'dq':env.d.qvel[env.s.dof].tolist(),'physics_time_s':float(env.d.time),'scenario':env.scenario}
 for k in range(len(ref.q)-1):
  action,_=policy.predict(obs,deterministic=True); obs,reward,term,trunc,info=env.step(action)
  qr,_,_=env._reference(); q=env.d.qpos[env.s.qadr].copy(); dq=env.d.qvel[env.s.dof].copy(); qp=info['qp']
  rows.append({'step':k+1,'control_time_s':(k+1)*DT,'physics_time_s':float(env.d.time),'q':q.tolist(),'dq':dq.tolist(),'qref':qr.tolist(),'ee_actual_m':fk(env,q).tolist(),'ee_reference_m':fk(env,qr).tolist(),'tracking_error_rad':float(np.sqrt(np.mean((q-qr)**2))),'action_normalized':np.asarray(action,float).tolist(),'raw_torque_nm':np.asarray(info['raw_torque_nm'],float).tolist(),'qp_torque_nm':np.asarray(info['safe_torque_nm'],float).tolist(),'projection_event':bool(info['projected']),'projection_norm_nm':float(qp.get('projection_norm_nm',0.0)),'qp_solver':qp.get('solver'),'termination_reason':info['termination_reason'],'monitoring':{'C_projection':float(info['C_projection']),'C_tau':float(info['C_tau'])}})
  if term or trunc: reason=info['termination_reason'] or ('horizon_complete' if trunc else 'terminated');break
 env.close()
 return rows,initial,reason

def visual_xml(path):
 s=(ROOT/'assets/panda_direct_torque_v2.xml').read_text().replace('meshdir="mujoco_menagerie/franka_emika_panda/assets"','meshdir="../../../../assets/mujoco_menagerie/franka_emika_panda/assets"')
 # Matte floor, no isolated disc, lower diffuse/ambient lighting: visual only.
 inject='''\n    <!-- V2-B2 visual-only objects: no collision or dynamics participation. -->
    <geom name="v2b2_floor_visual" type="plane" size="4 4 .1" pos="0 0 -.012" rgba="0.30 0.37 0.42 1" contype="0" conaffinity="0"/>
    <geom name="v2b2_base_visual" type="cylinder" size=".205 .018" pos="0 0 -.018" rgba=".13 .19 .23 1" contype="0" conaffinity="0"/>
    <light name="v2b2_key_visual" pos="1.2 -1.1 2.4" dir="-1 1 -2" directional="true" ambient=".25 .25 .25" diffuse=".50 .50 .50" specular=".10 .10 .10"/>
    <light name="v2b2_fill_visual" pos="-1.1 .8 1.6" dir="1 -.5 -1" directional="true" ambient=".04 .04 .04" diffuse=".18 .23 .28" specular=".03 .03 .03"/>\n'''
 path.write_text(s.replace('  </worldbody>',inject+'  </worldbody>'))
def backdrop(a):
 a=np.asarray(a).copy(); can=a.max(2)<10; seed=np.zeros_like(can);seed[0]=can[0];seed[-1]=can[-1];seed[:,0]|=can[:,0];seed[:,-1]|=can[:,-1];mask=binary_propagation(seed,mask=can); y=np.linspace(0,1,a.shape[0])[:,None];grad=np.dstack([29+13*y,42+16*y,52+19*y]).astype(np.uint8);a[mask]=np.broadcast_to(grad,a.shape)[mask];return a
def cam():
 c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.azimuth=132;c.elevation=-16;c.distance=1.50;c.lookat=np.array([.07,0,.48]);return c
def addcap(scene,p0,p1,w,col):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.ones(3),np.zeros(3),np.eye(3).ravel(),col);mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,w,np.asarray(p0),np.asarray(p1));scene.ngeom+=1
def addsphere(scene,p,r,col):
 if scene.ngeom>=scene.maxgeom:return
 mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([r]*3),p,np.eye(3).ravel(),col);scene.ngeom+=1
def panel(model,data,qadr,row,actual,ref,idx):
 data.qpos[qadr]=row['q'];data.qvel[:]=0;mujoco.mj_forward(model,data); ren=mujoco.Renderer(model,height=610,width=930,max_geom=1000);ren.update_scene(data,camera=cam())
 # Full future reference is low-alpha; travelled actual path is thin and opaque.
 pts=np.unique(np.linspace(0,len(ref)-1,min(100,len(ref)),dtype=int))
 for a,b in zip(pts[:-1],pts[1:]):addcap(ren.scene,ref[a],ref[b],.0023,AMBER)
 past=np.unique(np.linspace(0,idx,min(90,idx+1),dtype=int))
 for a,b in zip(past[:-1],past[1:]):addcap(ren.scene,actual[a],actual[b],.0035,CYAN)
 addsphere(ren.scene,ref[idx],.012,np.array([.96,.62,.13,1.]));addsphere(ren.scene,actual[idx],.011,np.array([.05,.78,.82,1.]))
 im=backdrop(ren.render());ren.close();return im
def line(draw,xy,box,values,lo,hi,color,width=3):
 if len(values)<2:return
 x0,y0,x1,y1=box; pts=[]
 for i,v in enumerate(values):pts.append((x0+(x1-x0)*i/(len(values)-1),y1-(v-lo)/(hi-lo)*(y1-y0)))
 draw.line(pts,fill=color,width=width)
def chart(draw,box,title,a,b,idx,colors,unit):
 x0,y0,x1,y1=box;draw.rounded_rectangle(box,12,fill=(12,25,34,245),outline=(76,109,126,180),width=2);draw.text((x0+18,y0+15),title,font=font(18,True),fill=(224,236,241));maxv=max(float(np.max(a)),float(np.max(b)),1e-5)*1.15;draw.text((x0+18,y0+45),f'0–{maxv:.3f} {unit}',font=font(14),fill=(146,176,190));plot=(x0+20,y0+78,x1-20,y1-25);line(draw,None,plot,a[:idx+1],0,maxv,colors[0]);line(draw,None,plot,b[:idx+1],0,maxv,colors[1]);x=plot[0]+(plot[2]-plot[0])*idx/(len(a)-1);draw.line((x,plot[1],x,plot[3]),fill=(235,244,248,160),width=2);return maxv
def compose(left,right,rows_a,rows_b,idx):
 im=Image.new('RGB',(W,H),(9,18,25));d=ImageDraw.Draw(im,'RGBA');d.rectangle((0,0,W,100),fill=(12,29,40));d.text((42,20),'FROZEN RESIDUAL SAC · DEMO-ONLY COMPARISON',font=font(29,True),fill=(235,243,247));d.text((42,59),'Multi-Pose Sweep · identical reference · seed 201 · exploratory replay',font=font(17),fill=(164,194,207));t=rows_a[idx]['control_time_s'];d.text((1600,31),f't = {t:4.2f} s',font=font(28,True),fill=(47,198,207));
 im.paste(Image.fromarray(left),(20,120));im.paste(Image.fromarray(right),(970,120));
 for x,label,c in [(20,LABELS['rbf_residual'],COLORS['rbf_residual']),(970,LABELS['pd_gravity_residual'],COLORS['pd_gravity_residual'])]:d.rounded_rectangle((x,112,x+400,153),10,fill=(12,25,34,232));d.rectangle((x+10,122,x+18,143),fill=c);d.text((x+30,119),label,font=font(17,True),fill=(235,242,246))
 ea=np.array([x['tracking_error_rad'] for x in rows_a]);eb=np.array([x['tracking_error_rad'] for x in rows_b]);pa=np.array([x['projection_norm_nm'] for x in rows_a]);pb=np.array([x['projection_norm_nm'] for x in rows_b]);chart(d,(38,770,938,1040),'INSTANTANEOUS JOINT TRACKING ERROR · RMS',ea,eb,idx,[COLORS['rbf_residual'],COLORS['pd_gravity_residual']],'rad');chart(d,(982,770,1882,1040),'QP COMMAND MODIFICATION · ||τraw − τQP||₂',pa,pb,idx,[COLORS['rbf_residual'],COLORS['pd_gravity_residual']],'Nm')
 d.text((48,735),'AMBER = reference FK   CYAN = actual FK   |   QP projection = command modification, not a safety violation.',font=font(16),fill=(188,211,221));d.text((48,1048),'RBF + PD + Residual SAC',font=font(15,True),fill=COLORS['rbf_residual']);d.text((295,1048),'PD + Gravity + Residual SAC',font=font(15,True),fill=COLORS['pd_gravity_residual']);return np.asarray(im)
def main():
 if OUT.exists():raise SystemExit('append-only output exists')
 OUT.mkdir(parents=True);probe=V2A.make_env('pd_gravity_residual',V2A.Reference('probe',np.zeros((2,7))));home=probe.home.copy();probe.close();ref=V2A.sweep(home)
 records={};initials={};reasons={}
 for mode in POLICIES:records[mode],initials[mode],reasons[mode]=replay(mode,ref);(OUT/f'{mode}_records.json').write_text(json.dumps(records[mode],indent=2)+'\n')
 if len(records['rbf_residual'])!=2375 or len(records['pd_gravity_residual'])!=2375:raise RuntimeError('fail closed: a required controller did not complete')
 if initials['rbf_residual']!=initials['pd_gravity_residual']:raise RuntimeError('fail closed: initial conditions differ')
 visual_xml(OUT/'studio_visual_only.xml');model=mujoco.MjModel.from_xml_path(str(OUT/'studio_visual_only.xml'));model.vis.global_.offwidth=930;model.vis.global_.offheight=610;qadr=np.array([model.joint(f'joint{i}').qposadr[0] for i in range(1,8)]);data=mujoco.MjData(model)
 ra,rb=records['rbf_residual'],records['pd_gravity_residual'];n=round(9.5*FPS);sample_times=np.arange(n)/FPS;indices=np.rint(sample_times/DT).astype(int);indices=np.minimum(indices,len(ra)-1)
 mapping=[];writer=imageio.get_writer(OUT/'v2b2_frozen_sac_synchronized_comparison.mp4',fps=FPS,codec='libx264',quality=8,macro_block_size=1)
 for f,i in enumerate(indices):
  left=panel(model,data,qadr,ra[i],np.array([z['ee_actual_m'] for z in ra]),np.array([z['ee_reference_m'] for z in ra]),i);right=panel(model,data,qadr,rb[i],np.array([z['ee_actual_m'] for z in rb]),np.array([z['ee_reference_m'] for z in rb]),i);writer.append_data(compose(left,right,ra,rb,i));mapping.append({'frame':f,'video_time_s':f/FPS,'control_index':int(i),'control_time_s':ra[i]['control_time_s'],'rbf_step':ra[i]['step'],'pd_gravity_step':rb[i]['step'],'rbf_projection_event':ra[i]['projection_event'],'pd_gravity_projection_event':rb[i]['projection_event']})
 writer.close();
 with open(OUT/'V2B2_FRAME_TIME_MAPPING.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=mapping[0]);w.writeheader();w.writerows(mapping)
 # Highlight: 4.5–9.5s has sustained divergence in real QP activity.
 hi=[x for x in mapping if 4.5<=x['video_time_s']<9.5];r=imageio.get_reader(OUT/'v2b2_frozen_sac_synchronized_comparison.mp4');wr=imageio.get_writer(OUT/'v2b2_comparison_highlight.mp4',fps=FPS,codec='libx264',quality=8,macro_block_size=1)
 for x in hi:wr.append_data(r.get_data(x['frame']))
 r.close();wr.close()
 frames=imageio.get_reader(OUT/'v2b2_frozen_sac_synchronized_comparison.mp4');chosen=[0,71,142,213,284];thumb=Image.new('RGB',(1920,1620),(9,18,25))
 for j,i in enumerate(chosen):thumb.paste(Image.fromarray(frames.get_data(i)).resize((960,540)),((j%2)*960,(j//2)*540))
 frames.close();thumb.save(OUT/'full_timeline_contact_sheet.png');Image.fromarray(imageio.get_reader(OUT/'v2b2_frozen_sac_synchronized_comparison.mp4').get_data(142)).save(OUT/'same_time_keyframe.png')
 # Static, independently reproducible plots and compact metrics.
 import matplotlib.pyplot as plt
 t=np.array([x['control_time_s'] for x in ra]);ea=np.array([x['tracking_error_rad'] for x in ra]);eb=np.array([x['tracking_error_rad'] for x in rb]);pa=np.array([x['projection_norm_nm'] for x in ra]);pb=np.array([x['projection_norm_nm'] for x in rb]);fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True);ax[0].plot(t,ea,label='RBF + PD + SAC');ax[0].plot(t,eb,label='PD + gravity + SAC');ax[0].set_ylabel('Joint RMS error (rad)');ax[0].legend();ax[1].plot(t,pa,label='RBF QP norm');ax[1].plot(t,pb,label='PD+gravity QP norm');ax[1].set_ylabel('||raw − QP||₂ (Nm)');ax[1].set_xlabel('Simulation time (s)');ax[1].legend();fig.tight_layout();fig.savefig(OUT/'tracking_and_qp_time_series.png',dpi=160);plt.close(fig)
 summary={}
 for mode,rs in records.items():
  a=np.array([z['ee_actual_m'] for z in rs]);e=np.array([z['tracking_error_rad'] for z in rs]);pn=np.array([z['projection_norm_nm'] for z in rs]);summary[mode]={'label':LABELS[mode],'policy_path':str(POLICIES[mode].relative_to(ROOT)),'policy_sha256':sha(POLICIES[mode]),'prior_path':'models/priors/rbf_w3.npz' if mode=='rbf_residual' else None,'prior_sha256':sha(ROOT/'models/priors/rbf_w3.npz') if mode=='rbf_residual' else None,'steps':len(rs),'duration_s':len(rs)*DT,'termination_reason':reasons[mode],'ee_path_length_m':float(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()),'mean_joint_rms_rad':float(e.mean()),'max_joint_rms_rad':float(e.max()),'projection_fraction':float(np.mean([z['projection_event'] for z in rs])),'mean_projection_norm_nm':float(pn.mean()),'max_projection_norm_nm':float(pn.max())}
 manifest={'classification':'DEMO_ONLY / EXPLORATORY_VISUALIZATION','replay_attempt':'new deterministic B2 frozen-policy replay; V2-A had no complete policy state records','reference':'V2-A Multi-Pose Sweeping exact definition','physics_dt_s':.002,'control_dt_s':DT,'fps':FPS,'frame_sampling':'frame f samples saved state at control index round((f/30)/0.004); f=0..284, so final frame time is 9.4666667 s rather than an erroneous 9.5 s endpoint','initial_state':initials['rbf_residual'],'controller_summaries':summary,'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()

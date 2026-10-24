"""Create labeled replay-only 1080p demonstrations from already exposed P5 scenes.

This script deliberately uses frozen policies and P5 scenario IDs 40000--40029.
It neither writes evaluation CSVs nor participates in P4/P5 selection/statistics.
"""
import csv,hashlib,json,os
from pathlib import Path
import imageio.v2 as imageio
import mujoco, numpy as np
from PIL import Image,ImageDraw,ImageFont
from stable_baselines3 import SAC
import p5_nominal_isolated_confirmation as strict
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'r3_mujoco/final_consolidation/visualization/high_quality_demos_attempt02'
P4=ROOT/'r3_mujoco/p4b_full_attempt02';PDG=ROOT/'r3_mujoco/p4b_pd_gravity_attempt02'
ARMS=(('MLP + PD + Residual SAC','mlp_residual',P4,'CONFIRMATORY P5 ARM'),('Expanded GP + PD + Residual SAC','gp_residual',P4,'CONFIRMATORY P5 ARM'),('Expanded RBF + PD + Residual SAC','rbf_residual',P4,'CONFIRMATORY P5 ARM'),('PD + Nominal Gravity + Residual SAC','pd_gravity_residual',PDG,'POSTHOC EXPLORATORY BASELINE'))
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def font(n,bold=False):
 p='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else FONT
 return ImageFont.truetype(p,n) if Path(p).exists() else ImageFont.load_default()
def overlay(arr,lines,fail=False):
 im=Image.fromarray(arr).convert('RGB');d=ImageDraw.Draw(im,'RGBA');d.rectangle((0,0,arr.shape[1],76),fill=(0,0,0,175));d.rectangle((0,76,arr.shape[1],80),fill=(220,45,45,255) if fail else (33,180,135,255))
 y=7
 for i,x in enumerate(lines):d.text((10,y),x,font=font(23 if i==0 else 17,bold=i==0),fill='white');y+=27
 return np.asarray(im)
def render_arm(label,mode,root,scene_seed,condition):
 model_path=root/mode/'seed_201'/'best_model.zip';model=SAC.load(model_path,device='cpu');env=strict.make(mode,condition,'final',scene_seed);obs,_=env.reset(seed=scene_seed)
 # The frozen MJCF advertises a 640-pixel offscreen framebuffer. Render at
 # its native safe size and upscale only for presentation encoding.
 renderer=mujoco.Renderer(env.m,height=480,width=640);cam=mujoco.MjvCamera();cam.azimuth=145;cam.elevation=-22;cam.distance=3.0;cam.lookat=np.array([0.,0.,.42])
 frames=[];metrics=[];why=''
 for k in range(500):
  action,_=model.predict(obs,deterministic=True);obs,reward,term,trunc,info=env.step(action);q=env.d.qpos[env.s.qadr];qr,_,_=env._reference();err=float(np.sqrt(np.mean((q-qr)**2)))
  metrics.append({'control_step':k+1,'sim_time_s':(k+1)*.004,'q_rmse_instant_rad':err,'projected':int(info['projected']),'termination_reason':info['termination_reason']})
  if k%8==0 or term or trunc:
   renderer.update_scene(env.d,camera=cam);im=np.asarray(Image.fromarray(renderer.render()).resize((960,540),Image.Resampling.LANCZOS));lines=[label,f"{env.scenario['task']} | seed {scene_seed} | t={(k+1)*.004:.3f} s",f"q error {err:.4f} rad | QP projected: {bool(info['projected'])}"]
   if term:lines.append('SAFETY TERMINATION: '+info['termination_reason'])
   frames.append(overlay(im,lines,fail=bool(term)))
  if term or trunc:why=info['termination_reason'];break
 renderer.close();return frames,metrics,{'label':label,'mode':mode,'model_path':str(model_path.relative_to(ROOT)),'model_sha256':digest(model_path),'steps':k+1,'termination_reason':why,'condition':condition['id'],'scene':env.scenario}
def grid(frames_by_arm,meta):
 n=max(len(x) for x in frames_by_arm)
 out=[]
 for i in range(n):
  cells=[x[min(i,len(x)-1)] for x in frames_by_arm];canvas=np.zeros((1080,1920,3),dtype=np.uint8)
  canvas[:540,:960]=cells[0];canvas[:540,960:]=cells[1];canvas[540:,:960]=cells[2];canvas[540:,960:]=cells[3]
  im=Image.fromarray(canvas);d=ImageDraw.Draw(im,'RGBA');d.rectangle((0,0,1920,42),fill=(12,20,34,210));d.text((14,8),meta,font=font(22,True),fill=(255,255,255));out.append(np.asarray(im))
 return out
def video(path,frames,fps=30):
 with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=1) as w:
  for x in frames:w.append_data(x)
def gif(path,frames):
 Image.fromarray(frames[0]).save(path,save_all=True,append_images=[Image.fromarray(x) for x in frames[1:]],duration=33,loop=0,optimize=False)
def mismatch_frames(scene_seed):
 # RBF is selected because persisted strict replay records a representative
 # combined-strong failure at this exposed point-to-point scene.
 arm=('Expanded RBF + PD + Residual SAC','rbf_residual',P4,'CONFIRMATORY P5 ARM');nom= strict.cfg()['conditions'][0];mis=[x for x in strict.cfg()['conditions'] if x['id']=='combined_strong'][0]
 nf,nm,nmeta=render_arm(*arm[:3],scene_seed,nom);mf,mm,mmeta=render_arm(*arm[:3],scene_seed,mis);n=max(len(nf),len(mf));out=[]
 for i in range(n):
  # Upscale two independent 16:9 panes to the 1080p split-screen height.
  left=np.asarray(Image.fromarray(nf[min(i,len(nf)-1)]).resize((960,1080),Image.Resampling.LANCZOS));right=np.asarray(Image.fromarray(mf[min(i,len(mf)-1)]).resize((960,1080),Image.Resampling.LANCZOS))
  c=np.zeros((1080,1920,3),dtype=np.uint8);c[:,:960]=left;c[:,960:]=right
  im=Image.fromarray(c);d=ImageDraw.Draw(im,'RGBA');d.rectangle((0,0,1920,44),fill=(90,25,25,215));d.text((14,9),'EXPOSED P5 REPLAY — Nominal (left) vs Combined-Strong mismatch (right); no new evaluation',font=font(20,True),fill='white');out.append(np.asarray(im))
 return out,{'nominal':nmeta,'mismatch':mmeta,'nominal_metrics':nm,'mismatch_metrics':mm}
def title_frame(text,sub='REPLAY-ONLY VISUALIZATION — NOT A NEW EVALUATION'):
 x=np.zeros((1080,1920,3),dtype=np.uint8);x[:]=[15,25,42];im=Image.fromarray(x);d=ImageDraw.Draw(im);d.text((110,430),text,font=font(54,True),fill=(240,245,250));d.text((110,520),sub,font=font(26),fill=(130,190,210));return np.asarray(im)
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir(parents=True);manifest={'classification':'EXPOSED_SCENARIO_REPLAY_DIAGNOSTIC_ONLY','fps':30,'resolution':'1920x1080','control_dt_s':.004,'physics_dt_s':.002,'clips':[],'prohibitions':['not P4/P5 evidence','not used for model/checkpoint/controller selection','not real robot evidence']}
 clips=[]
 for name,seed in [('point_to_point',40023),('multi_joint',40019),('single_joint_auxiliary',40015),('multi_joint_alt',40025)]:
  condition=strict.cfg()['conditions'][0]; rendered=[];per=[]
  for a in ARMS:
   f,m,z=render_arm(*a[:3],seed,condition);rendered.append(f);per.append({**z,'metrics':m})
  fr=grid(rendered,f'EXPOSED P5 NOMINAL REPLAY | {name} | common seed {seed} | 2 s tracking')
  mp4=OUT/f'four_controller_sync_{name}_seed{seed}.mp4';video(mp4,fr);gif(OUT/f'four_controller_sync_{name}_seed{seed}_readme.gif',fr)
  clips.append((name,mp4));manifest['clips'].append({'kind':'four_controller_sync','scene_seed':seed,'task':name,'condition':'nominal','video':mp4.name,'arms':per})
 mismatch,md=mismatch_frames(40023);mp4=OUT/'nominal_vs_combined_strong_rbf_seed40023.mp4';video(mp4,mismatch);gif(OUT/'nominal_vs_combined_strong_rbf_seed40023_readme.gif',mismatch);clips.append(('mismatch',mp4));manifest['clips'].append({'kind':'mismatch','scene_seed':40023,'condition':'combined_strong','video':mp4.name,**md})
 # 16-second reel: title + four exact 2-second nominal fragments + mismatch + closing disclosure.
 reel=[title_frame('Franka Panda: Safety-Constrained Residual SAC') for _ in range(30)]
 for name,p in clips:
  rdr=imageio.get_reader(p)
  reel.extend([x for x in rdr]);rdr.close()
  reel.extend([title_frame('Next clip: '+name,'Each fragment is an explicitly labeled, exposed-scenario replay.') for _ in range(15)])
 reel.extend([title_frame('Simulation evidence only','Strict P5 has low all-condition completion; see Final Research Report.') for _ in range(30)])
 video(OUT/'main_showcase_reel.mp4',reel);Image.fromarray(reel[30]).save(OUT/'cover.png')
 # Compact timeline for reproducibility/audit.
 with (OUT/'SCENE_SELECTION.json').open('w') as f:json.dump(manifest,f,indent=2)
 (OUT/'README.md').write_text('# High-quality MuJoCo demonstrations\n\nAll clips replay already exposed P5 scenes with frozen seed-201 policies. They are visual diagnostics only, not new evaluation units. Four-controller grids use one camera, one scenario, and one time base. PD+Gravity is visibly labeled posthoc exploratory.\n')
if __name__=='__main__':main()

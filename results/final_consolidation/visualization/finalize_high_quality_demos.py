"""Finalize the append-only high-quality replay reel after its component clips exist."""
import hashlib,json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'r3_mujoco/final_consolidation/visualization/high_quality_demos_attempt02'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def title(text,sub):
 from PIL import ImageDraw,ImageFont
 x=np.zeros((1080,1920,3),dtype=np.uint8);x[:]=[15,25,42];im=Image.fromarray(x);d=ImageDraw.Draw(im)
 fp='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';fn='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 f1=ImageFont.truetype(fp,54);f2=ImageFont.truetype(fn,26)
 d.text((110,430),text,font=f1,fill=(240,245,250));d.text((110,520),sub,font=f2,fill=(130,190,210));return np.asarray(im)
def main():
 reel=OUT/'main_showcase_reel.mp4'
 if reel.exists():raise RuntimeError('reel exists')
 clips=[('Point-to-point — common exposed seed 40023',OUT/'four_controller_sync_point_to_point_seed40023.mp4'),('Multi-joint — common exposed seed 40019',OUT/'four_controller_sync_multi_joint_seed40019.mp4'),('Single-joint auxiliary — common exposed seed 40015',OUT/'four_controller_sync_single_joint_auxiliary_seed40015.mp4'),('Multi-joint alternate — common exposed seed 40025',OUT/'four_controller_sync_multi_joint_alt_seed40025.mp4'),('RBF nominal versus combined-strong mismatch',OUT/'nominal_vs_combined_strong_rbf_seed40023.mp4')]
 frames=[title('Franka Panda: Safety-Constrained Residual SAC','EXPOSED-SCENARIO REPLAY ONLY — NOT A NEW EVALUATION') for _ in range(45)]
 for name,p in clips:
  r=imageio.get_reader(p);frames.extend([np.asarray(x) for x in r]);r.close()
  frames.extend([title('Next: '+name,'Frozen seed-201 policies; 4 ms control / 2 ms physics; P5 metrics remain unchanged.') for _ in range(15)])
 frames.extend([title('Simulation evidence only','PD+Gravity is posthoc exploratory. Mismatch failure is retained, not hidden.') for _ in range(45)])
 with imageio.get_writer(reel,fps=30,codec='libx264',quality=8,macro_block_size=1) as w:
  for x in frames:w.append_data(x)
 Image.fromarray(frames[45]).save(OUT/'cover.png')
 # Small, README-suitable GIF: short point-to-point comparator; it does not alter the MP4 timeline.
 src=imageio.get_reader(clips[0][1]);small=[]
 for i,x in enumerate(src):
  if i%2==0:small.append(np.asarray(Image.fromarray(x).resize((960,540),Image.Resampling.LANCZOS)))
 src.close();Image.fromarray(small[0]).save(OUT/'github_preview_point_to_point.gif',save_all=True,append_images=[Image.fromarray(x) for x in small[1:]],duration=67,loop=0,optimize=False)
 data={'classification':'EXPOSED_SCENARIO_REPLAY_DIAGNOSTIC_ONLY','renderer':'MuJoCo EGL; frozen 640x480 framebuffer upscaled for 1080p encoding','fps':30,'main_reel_seconds':len(frames)/30,'selection':[{'seed':40023,'task':'point_to_point','reason':'largest exposed point-to-point range among selected successful frozen replays'},{'seed':40019,'task':'multi_joint','reason':'large coordinated seven-joint range'},{'seed':40015,'task':'single_joint','reason':'auxiliary J4 motion'},{'seed':40025,'task':'multi_joint','reason':'second multi-joint fragment'},{'seed':40023,'condition':'combined_strong','reason':'persisted RBF safety termination demonstration'}],'clips':[{'file':p.name,'sha256':sha(p)} for _,p in clips]+[{'file':reel.name,'sha256':sha(reel)}],'science_boundary':['all P5 scene IDs were already exposed','no P4/P5 statistical result was recomputed or added','PD+Gravity is posthoc exploratory','rendered state is not altered']}
 (OUT/'SCENE_SELECTION.json').write_text(json.dumps(data,indent=2)+'\n')
 (OUT/'README.md').write_text('# High-quality MuJoCo control demonstrations\n\n`main_showcase_reel.mp4` is a 1080p/30fps sequence of explicitly separated 2-second, exposed-scenario replays. Four-controller clips use a common seed, camera, and time base. `nominal_vs_combined_strong_rbf_seed40023.mp4` retains a real safety termination. These visual diagnostics are not P4/P5 evaluation data.\n')
 # Decode checks: every final MP4 must yield 1920x1080 RGB frames.
 checks=[]
 for _,p in clips+[('main',reel)]:
  r=imageio.get_reader(p);a=np.asarray(r.get_data(0));meta=r.get_meta_data();r.close()
  if a.shape!=(1080,1920,3):raise RuntimeError('bad video shape '+str(p)+str(a.shape))
  checks.append({'file':p.name,'first_frame_shape':list(a.shape),'fps':meta.get('fps'),'sha256':sha(p)})
 (OUT/'VIDEO_QUALITY_CHECK.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
if __name__=='__main__':main()

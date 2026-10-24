#!/usr/bin/env python3
"""V2-C local final-edit assembly from audited, immutable source media."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;OUT=B/'outputs_v2c'/'attempt_01';FPS=30;SZ=(1920,1080)
S={'b1':B/'outputs_v2b1/attempt_04/v2b1_studio_motion_preview.mp4','b2':B/'outputs_v2b2_repair/attempt_01/v2b2r_frozen_sac_synchronized_comparison.mp4','b3':B/'outputs_v2b3/attempt_01/v2b3_authentic_acceleration_limit_failure_extended.mp4','p4':ROOT/'docs/assets/figures/p4b_validation_six_group_summary.png','p5':ROOT/'docs/assets/figures/p5_strict_completion_heatmap.png','p5o':ROOT/'docs/assets/figures/p5_overall_completion_strict_vs_posthoc.png'}
def sh(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def ft(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def badge(a,title,sub):
 im=Image.fromarray(a).convert('RGB');d=ImageDraw.Draw(im,'RGBA');d.rounded_rectangle((28,25,1000,104),16,fill=(8,22,31,210));d.text((53,39),title,font=ft(24,True),fill=(235,244,247));d.text((53,72),sub,font=ft(15),fill=(177,205,216));return np.asarray(im)
def video(path,title,sub):
 r=iio.get_reader(path);out=[]
 for x in r:out.append(badge(np.asarray(Image.fromarray(x).resize(SZ)),title,sub))
 r.close();return out
def still(path,title,sub,seconds=7):
 im=Image.open(path).convert('RGB');im.thumbnail((1700,800));can=Image.new('RGB',SZ,(12,25,34));can.paste(im,((1920-im.width)//2,150));d=ImageDraw.Draw(can,'RGBA');d.rectangle((0,0,1920,126),fill=(10,26,37));d.text((48,30),title,font=ft(30,True),fill=(235,244,247));d.text((48,76),sub,font=ft(18),fill=(178,205,216));d.text((48,1030),'Frozen formal result figure · see report for protocol and limits',font=ft(16),fill=(170,196,207));return [np.asarray(can)]*(seconds*FPS)
def write(name,frames):
 w=iio.get_writer(OUT/name,fps=FPS,codec='libx264',quality=8,macro_block_size=1)
 for x in frames:w.append_data(x)
 w.close()
def main():
 if OUT.exists():raise SystemExit('append-only output exists')
 OUT.mkdir(parents=True)
 b1=video(S['b1'],'DEMO ONLY · Multi-Pose Sweep','Recorded PD + Gravity · zero SAC residual · 1× simulation time')
 b2=video(S['b2'],'DEMO ONLY · Frozen residual-SAC comparison','Single trajectory · seed 201 · QP projection is command modification, not a safety violation')
 b3=video(S['b3'],'DEMO ONLY · Authentic fail-closed case','Frozen PD + Residual SAC · nominal plant · final section contains a labelled freeze frame')
 p4=still(S['p4'],'P4 validation · six controller groups','Frozen report figure; controller results are not inferred from showcase media',7)
 p5=still(S['p5'],'P5 mismatch robustness · strict completion','Frozen report figure; PD + Gravity posthoc results remain exploratory',7)
 p5o=still(S['p5o'],'Research scope and reproducibility','Code, reports, configurations and limitations: github.com/Eurekauro/franka-dynamics-priors-residual-sac',7)
 # Highlight starts directly with physical motion. No title card precedes it.
 write('v2c_github_highlight_reel.mp4',b1+b2+b3+p5o[:3*FPS])
 write('v2c_research_demonstration.mp4',b1+video(S['b1'],'Dynamics priors + residual control','Reference amber · actual cyan · visual replay uses recorded states')+b2+p4+p5+b3+p5o)
 # 5-second README GIF from real synchronized motion only.
 r=iio.get_reader(S['b2']);g=[Image.fromarray(r.get_data(i)) for i in range(135,285,2)];r.close();g[0].save(OUT/'v2c_readme_motion.gif',save_all=True,append_images=g[1:],duration=67,loop=0,optimize=False)
 Image.fromarray(b2[142]).save(OUT/'v2c_cover.png')
 storyboard='''# V2-C storyboard\n\n- Highlight (about 26.6 s): immediate Multi-Pose motion; B2-R synchronized frozen-policy comparison; authentic B3 nominal-plant acceleration-limit termination; three-second frozen-result scope card.\n- Research demonstration (about 68.6 s): recorded Studio movement and method context; B2-R comparison; frozen P4/P5 figures; B3 authentic termination including labelled freeze; scope/repository close.\n- All movement is source media at its encoded 1× cadence. Static research figures are explicitly frozen report figures; episode changes are badged.\n''';(OUT/'STORYBOARD.md').write_text(storyboard)
 man={'classification':'DEMO_ONLY / EXPLORATORY_VISUALIZATION','sources':{k:{'path':str(v.relative_to(ROOT)),'sha256':sh(v)} for k,v in S.items()},'motion_integrity':'No controller or physics replay. Video clips are frame-preserving source-media edits; B3 freeze is pre-labelled source media.','files':{p.name:sh(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}};(OUT/'MANIFEST.json').write_text(json.dumps(man,indent=2)+'\n')
if __name__=='__main__':main()

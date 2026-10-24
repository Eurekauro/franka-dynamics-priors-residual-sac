#!/usr/bin/env python3
"""Deterministic V2-E-R edit from audited media; no physics/controller execution."""
from pathlib import Path
import json,hashlib
import imageio.v2 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;O=B/'outputs_v2e_final'/'attempt_02';S=R/'results/final_consolidation/visualization/high_quality_demos_attempt02';F=R/'docs/assets/figures';FPS=30;WH=(1920,1080)
def ft(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def sh(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def title(im,headline,detail):
 d=ImageDraw.Draw(im,'RGBA');d.rounded_rectangle((32,26,1500,112),18,fill=(9,27,39,224));d.text((58,43),headline,font=ft(25,True),fill=(237,245,248));d.text((58,76),detail,font=ft(16),fill=(182,210,220));return im
def single(path,label):
 r=iio.get_reader(path);out=[]
 for a in r:
  # Existing four-arm replay: retain only upper-left frozen controller pane.
  x=Image.fromarray(a).crop((0,0,960,540)).resize((1480,833),Image.Resampling.LANCZOS);can=Image.new('RGB',WH,(24,40,53));can.paste(x,(220,165));title(can,label,'P5 Protocol-Matched Visualization Replay · 0.5× playback · simulated time remains 1× in source HUD');out.extend([np.asarray(can),np.asarray(can)])
 r.close();return out
def dual(path,label):
 r=iio.get_reader(path);out=[]
 for a in r:
  x=Image.fromarray(a).resize(WH,Image.Resampling.LANCZOS);title(x,label,'Same frozen RBF policy, reference and initial condition · P5 Protocol-Matched Visualization Replay · 0.5×');out.extend([np.asarray(x),np.asarray(x)])
 r.close();return out
def figure(path,head,sub,sec):
 x=Image.open(path).convert('RGB');x.thumbnail((1650,800));can=Image.new('RGB',WH,(24,40,53));can.paste(x,((1920-x.width)//2,170));title(can,head,sub);d=ImageDraw.Draw(can);d.text((46,1030),'Frozen research result figure · P4 validation is checkpoint-selection evidence, not untouched final testing.',font=ft(15),fill=(180,206,217));return [np.asarray(can)]*(sec*FPS)
def write(n,frames,size=WH):
 w=iio.get_writer(O/n,fps=FPS,codec='libx264',quality=8,macro_block_size=1)
 for a in frames:w.append_data(np.asarray(Image.fromarray(a).resize(size,Image.Resampling.LANCZOS)))
 w.close()
def main():
 if O.exists():raise SystemExit('append-only')
 O.mkdir(parents=True)
 multi=single(S/'four_controller_sync_multi_joint_seed40019.mp4','P5 nominal · multi-joint · frozen controller replay')
 point=single(S/'four_controller_sync_point_to_point_seed40023.mp4','P5 nominal · point-to-point · frozen controller replay')
 joint=single(S/'four_controller_sync_single_joint_auxiliary_seed40015.mp4','P5 nominal · single-joint · frozen controller replay')
 mismatch=dual(S/'nominal_vs_combined_strong_rbf_seed40023.mp4','P5 nominal (left) vs combined-strong forward-plant mismatch (right)')
 p4=figure(F/'p4b_validation_six_group_summary.png','P4 validation · six frozen controller groups','360 validation episodes · 12 scenario IDs / 9 unique references',8)
 learn=figure(F/'p4b_validation_learning_curves.png','P4 validation learning curves','Frozen validation figure; no P4 per-state replay is retained',8)
 p5=figure(F/'p5_strict_completion_heatmap.png','P5 confirmatory evaluation · 13 frozen dynamics conditions','Five confirmatory policy groups · 9,750 episodes',9)
 term=figure(F/'p5_termination_reasons.png','P5 fail-closed monitoring outcomes','QP command projection is not a safety violation',8)
 scope=figure(F/'p5_overall_completion_strict_vs_posthoc.png','Scope and limitation','PD + Gravity is posthoc exploratory; it is not part of the five-group confirmatory statistic',8)
 highlight=multi+point+mismatch+p4[:4*FPS]+p5[:4*FPS]+joint
 research=p4+multi+learn+point+joint+p5+mismatch+term+scope
 # Required low-resolution layout QA preview precedes high-resolution output.
 write('v2er_layout_preview.mp4',highlight,(960,540));write('v2er_github_highlight.mp4',highlight);write('v2er_research_demonstration.mp4',research)
 gif=multi[:150];Image.fromarray(gif[0]).resize((960,540),Image.Resampling.LANCZOS).save(O/'v2er_readme.gif',save_all=True,append_images=[Image.fromarray(a).resize((960,540),Image.Resampling.LANCZOS) for a in gif[1:]],duration=33,loop=0,optimize=True)
 Image.fromarray(highlight[0]).save(O/'v2er_cover.png');(O/'STORYBOARD.md').write_text('# V2-E-R\n\nHighlight: single-frame P5 nominal multi-joint/point-to-point; paired nominal-vs-combined-strong mismatch; frozen P4/P5 evidence; single-joint closing. Research Demo: frozen P4 figures and all three P5 task replays, then the sole dual-screen mismatch comparison and P5 frozen result figures. All dynamic fragments are labelled protocol-matched replay at 0.5×.\n')
 src=[S/'four_controller_sync_multi_joint_seed40019.mp4',S/'four_controller_sync_point_to_point_seed40023.mp4',S/'four_controller_sync_single_joint_auxiliary_seed40015.mp4',S/'nominal_vs_combined_strong_rbf_seed40023.mp4',F/'p4b_validation_six_group_summary.png',F/'p4b_validation_learning_curves.png',F/'p5_strict_completion_heatmap.png',F/'p5_termination_reasons.png',F/'p5_overall_completion_strict_vs_posthoc.png'];(O/'MANIFEST.json').write_text(json.dumps({'classification':'P5 Protocol-Matched Visualization Replay / frozen result presentation','source_sha256':{str(p.relative_to(R)):sh(p) for p in src},'render':'offline crop/scale/HUD composite only; no physics or policy execution','files':{p.name:sh(p) for p in O.iterdir() if p.is_file() and p.name!='MANIFEST.json'}},indent=2)+'\n')
if __name__=='__main__':main()

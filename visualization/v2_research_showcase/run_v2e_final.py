#!/usr/bin/env python3
"""V2-E formal-results-first edit from exposed P5 protocol-matched replays."""
from pathlib import Path
import imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont
import numpy as np,json,hashlib
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;O=B/'outputs_v2e'/'attempt_01';S=R/'results/final_consolidation/visualization/high_quality_demos_attempt02';F=R/'docs/assets/figures';FPS=30
def ft(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def sh(p):h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def clip(p,title,slow=False):
 r=iio.get_reader(p);out=[]
 for x in r:
  im=Image.fromarray(x);d=ImageDraw.Draw(im,'RGBA');d.rounded_rectangle((24,22,1250,92),14,fill=(8,23,32,210));d.text((45,37),title+(' · 0.5× playback' if slow else ' · 1× simulation'),font=ft(21,True),fill=(236,245,248));a=np.asarray(im);out.extend([a,a] if slow else [a])
 r.close();return out
def fig(p,title,sub,sec):
 im=Image.open(p).convert('RGB');im.thumbnail((1700,820));can=Image.new('RGB',(1920,1080),(11,24,33));can.paste(im,((1920-im.width)//2,150));d=ImageDraw.Draw(can);d.rectangle((0,0,1920,125),fill=(9,26,38));d.text((42,27),title,font=ft(29,True),fill=(239,245,248));d.text((42,76),sub,font=ft(18),fill=(181,207,218));return [np.asarray(can)]*(sec*FPS)
def write(n,x):
 w=iio.get_writer(O/n,fps=FPS,codec='libx264',quality=8,macro_block_size=1)
 for a in x:w.append_data(a)
 w.close()
def main():
 if O.exists():raise SystemExit('append-only')
 O.mkdir(parents=True)
 multi=clip(S/'four_controller_sync_multi_joint_seed40019.mp4','P5 Protocol-Matched Visualization Replay · nominal · multi-joint',True)
 point=clip(S/'four_controller_sync_point_to_point_seed40023.mp4','P5 Protocol-Matched Visualization Replay · nominal · point-to-point',True)
 single=clip(S/'four_controller_sync_single_joint_auxiliary_seed40015.mp4','P5 Protocol-Matched Visualization Replay · nominal · single-joint',True)
 mismatch=clip(S/'nominal_vs_combined_strong_rbf_seed40023.mp4','P5 Protocol-Matched Visualization Replay · RBF seed 201 · nominal vs combined-strong mismatch',True)
 p4=fig(F/'p4b_validation_six_group_summary.png','P4 Validation · 360 checkpoint-selection episodes','12 scenario IDs / 9 unique references · validation, not untouched final test',6)
 p5=fig(F/'p5_strict_completion_heatmap.png','P5 Confirmatory Evaluation · five frozen controller groups','9,750 episodes across 13 frozen dynamics conditions',6)
 lim=fig(F/'p5_termination_reasons.png','Safety interpretation and limits','QP projection is command modification; acceleration-limit is a monitored plant termination',5)
 out=fig(F/'p5_overall_completion_strict_vs_posthoc.png','Research conclusion and scope','PD + Gravity P5 results are posthoc exploratory and excluded from the five-group confirmatory statistic',5)
 write('v2e_github_highlight_reel.mp4',multi+point+mismatch+p4+p5[:2*FPS])
 write('v2e_research_demonstration.mp4',p4+multi+point+single+p5+mismatch+lim+out)
 g=multi[:150];Image.fromarray(g[0]).save(O/'v2e_readme_formal_motion.gif',save_all=True,append_images=[Image.fromarray(x) for x in g[1:]],duration=33,loop=0,optimize=False)
 Image.fromarray(multi[30]).save(O/'v2e_cover.png');(O/'STORYBOARD.md').write_text('# V2-E\n\nFormal-results-first edit. P4 is shown as frozen validation results because no source state trajectory is archived. Dynamic footage consists only of exposed P5 protocol-matched visualization replays, labelled with condition and 0.5× playback.\n')
 src=[S/'four_controller_sync_multi_joint_seed40019.mp4',S/'four_controller_sync_point_to_point_seed40023.mp4',S/'four_controller_sync_single_joint_auxiliary_seed40015.mp4',S/'nominal_vs_combined_strong_rbf_seed40023.mp4',F/'p4b_validation_six_group_summary.png',F/'p5_strict_completion_heatmap.png',F/'p5_termination_reasons.png',F/'p5_overall_completion_strict_vs_posthoc.png'];(O/'MANIFEST.json').write_text(json.dumps({'classification':'P5 Protocol-Matched Visualization Replay / frozen-result presentation','source_hashes':{str(x.relative_to(R)):sh(x) for x in src},'p4_motion_limitation':'No archived P4 per-state episode trajectory found; P4 is represented only by frozen result figures.','files':{}},indent=2)+'\n')
if __name__=='__main__':main()

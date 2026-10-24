#!/usr/bin/env python3
"""Offline B2 repair: saved records only; no policy/environment execution."""
from __future__ import annotations
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np,imageio.v2 as iio
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
BASE=Path(__file__).resolve().parent; SRC=BASE/'outputs_v2b2'/'attempt_04'; OUT=BASE/'outputs_v2b2_repair'/'attempt_01'; FPS=30;W,H=1920,1080
def f(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),n)
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def graph(d,box,title,a,b,k,cols,unit):
 x0,y0,x1,y1=box;d.rounded_rectangle(box,12,fill=(10,23,32,255),outline=(85,120,139,220),width=2);d.text((x0+18,y0+13),title,font=f(21,True),fill=(235,243,246));mx=max(float(np.max(a)),float(np.max(b)),1e-4)*1.15;d.text((x0+18,y0+46),f'0–{mx:.3f} {unit}  · fixed 0–9.5 s axis',font=f(15),fill=(160,190,205));px=(x0+28,y0+80,x1-24,y1-28)
 for vals,c in [(a,cols[0]),(b,cols[1])]:
  ids=np.where(np.arange(len(vals))<=k)[0];pts=[(px[0]+(px[2]-px[0])*(.004*(i+1)/9.5),px[3]-(px[3]-px[1])*vals[i]/mx) for i in ids]
  if len(pts)>1:d.line(pts,fill=c,width=3)
 t=.004*(k+1);x=px[0]+(px[2]-px[0])*t/9.5;d.line((x,px[1],x,px[3]),fill=(245,248,250,230),width=2);return t
def main():
 if OUT.exists():raise SystemExit('append-only exists')
 OUT.mkdir(parents=True);a=json.loads((SRC/'rbf_residual_records.json').read_text());b=json.loads((SRC/'pd_gravity_residual_records.json').read_text());old=iio.get_reader(SRC/'v2b2_frozen_sac_synchronized_comparison.mp4');maprows=[];writer=iio.get_writer(OUT/'v2b2r_frozen_sac_synchronized_comparison.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1)
 ea=np.array([x['tracking_error_rad'] for x in a]);eb=np.array([x['tracking_error_rad'] for x in b]);pa=np.array([x['projection_norm_nm'] for x in a]);pb=np.array([x['projection_norm_nm'] for x in b])
 for kf in range(285):
  target=kf/FPS; candidates=np.array([x['control_time_s'] for x in a]);ix=int(np.argmin(abs(candidates-target))) if target>=.004 else 0;frame=old.get_data(kf);im=Image.fromarray(frame);d=ImageDraw.Draw(im,'RGBA');d.rectangle((0,750,W,H),fill=(8,18,25,255));t=graph(d,(38,778,938,1040),'INSTANTANEOUS JOINT TRACKING ERROR · RMS',ea,eb,ix,[(168,93,207),(40,177,191)],'rad');graph(d,(982,778,1882,1040),'QP COMMAND MODIFICATION · ||τraw − τQP||₂',pa,pb,ix,[(168,93,207),(40,177,191)],'Nm');d.text((42,1050),'Purple: RBF + PD + Residual SAC     Cyan: PD + Gravity + Residual SAC',font=f(16,True),fill=(210,227,233));d.text((1320,1050),'QP projection = command modification, not a safety violation.',font=f(15),fill=(180,205,215));writer.append_data(np.asarray(im));maprows.append({'frame_index':kf,'video_timestamp_s':target,'state_source_index':ix,'physical_simulation_timestamp_s':a[ix]['control_time_s'],'cursor_x_time_s':t,'last_plotted_source_timestamp_s':a[ix]['control_time_s'],'qp_event_source_index':ix,'rbf_qp_event':a[ix]['projection_event'],'pd_gravity_qp_event':b[ix]['projection_event']})
 writer.close();old.close()
 with open(OUT/'V2B2R_FRAME_TIME_MAPPING.csv','w',newline='') as h:w=csv.DictWriter(h,fieldnames=maprows[0]);w.writeheader();w.writerows(maprows)
 r=iio.get_reader(OUT/'v2b2r_frozen_sac_synchronized_comparison.mp4');wr=iio.get_writer(OUT/'v2b2r_comparison_highlight.mp4',fps=30,codec='libx264',quality=8,macro_block_size=1)
 for z in range(135,285):wr.append_data(r.get_data(z))
 wr.close();sheet=Image.new('RGB',(1920,1620));
 for j,z in enumerate([0,71,142,213,284]):sheet.paste(Image.fromarray(r.get_data(z)).resize((960,540)),((j%2)*960,(j//2)*540))
 r.close();sheet.save(OUT/'full_timeline_contact_sheet.png');
 (OUT/'render_config.json').write_text(json.dumps({'source_records':str(SRC),'camera':'unchanged B2 fixed camera','chart_axis_s':[0,9.5],'state_sampling':'nearest saved physical control timestamp; start is actual t=0.004 s, no t=0 fabrication','no_controller_execution':True},indent=2)+'\n')
 m={'classification':'DEMO_ONLY / EXPLORATORY_VISUALIZATION','source_record_hashes':{p.name:sha(p) for p in [SRC/'rbf_residual_records.json',SRC/'pd_gravity_residual_records.json']},'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}};(OUT/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()

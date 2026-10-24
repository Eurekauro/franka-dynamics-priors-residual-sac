"""Render exposed P5 scenario 40000 for presentation only; no new evaluation claim."""
import csv,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from stable_baselines3 import SAC
import p5_nominal_isolated_confirmation as strict
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'r3_mujoco/final_consolidation/visualization/diagnostic_rollouts_attempt02'
P4=ROOT/'r3_mujoco/p4b_full_attempt02';PDG=ROOT/'r3_mujoco/p4b_pd_gravity_attempt02'
ARMS=(('pd_gravity_residual',PDG),('mlp_residual',P4),('gp_residual',P4),('rbf_residual',P4))
def main():
 if OUT.exists():raise RuntimeError('append-only diagnostic output exists')
 OUT.mkdir(parents=True);cond=strict.cfg()['conditions'][0];seed=40000;rows=[];manifest={'classification':'EXPOSED_SCENARIO_DIAGNOSTIC_ONLY','scenario_seed':seed,'condition':cond['id'],'policies':[],'not_for_metric_or_selection':True}
 for mode,root in ARMS:
  m=SAC.load(root/mode/'seed_201'/'best_model.zip',device='cpu');e=strict.make(mode,cond,'final',seed);obs,_=e.reset(seed=seed);frames=[];reason=''
  try:
   import mujoco
   renderer=mujoco.Renderer(e.m,height=360,width=480)
  except Exception as x: raise RuntimeError('MuJoCo renderer unavailable: '+str(x))
  for k in range(500):
   a,_=m.predict(obs,deterministic=True);obs,r,term,trunc,info=e.step(a)
   if k%10==0:
    renderer.update_scene(e.d);frames.append(Image.fromarray(renderer.render()))
   q=e.d.qpos[e.s.qadr].copy();qr,_,_=e._reference();rows.append({'mode':mode,'control_step':k+1,'q_rmse_instant_rad':float(np.sqrt(np.mean((q-qr)**2))),'projected':int(info['projected']),'terminated':int(term),'termination_reason':info['termination_reason']})
   if term or trunc:reason=info['termination_reason'];break
  renderer.close();frames[0].save(OUT/f'{mode}_nominal_seed40000.gif',save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=False)
  manifest['policies'].append({'mode':mode,'seed':201,'steps':k+1,'termination_reason':reason,'frames':len(frames)})
 with (OUT/'rollout_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()

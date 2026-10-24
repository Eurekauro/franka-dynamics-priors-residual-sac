"""Posthoc-only P5 replay for the new PD+gravity residual baseline.

It intentionally reuses the already exposed Round-06 final scenario definition.
No result from this script may be used to choose a checkpoint or alter training.
"""
import csv,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from stable_baselines3 import SAC
import p5_nominal_isolated_confirmation as strict
ROOT=Path(__file__).resolve().parents[1]
P4=Path(os.environ.get('PDG_P4_OUTPUT_DIR',ROOT/'r3_mujoco/p4b_pd_gravity_attempt01'))
OUT=Path(os.environ.get('PDG_P5_OUTPUT_DIR',ROOT/'r3_mujoco/p5_pd_gravity_exploratory_attempt01'))
MODE='pd_gravity_residual';SEEDS=(201,202,203,204,205)
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');os.replace(t,p)
def lane(seed):
 import torch
 os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
 m=SAC.load(P4/MODE/f'seed_{seed:03d}'/'best_model.zip',device='cpu');rows=[];t=time.perf_counter()
 for c in strict.cfg()['conditions']:
  for scenario_seed in strict.cfg()['scenarios']['seeds']:
   rows.append(strict.ep(m,MODE,seed,c,scenario_seed))
 return seed,rows,{'seed':seed,'episodes':len(rows),'wall_s':time.perf_counter()-t}
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 if any(not (P4/MODE/f'seed_{s:03d}'/'best_model.zip').exists() for s in SEEDS):raise RuntimeError('not all five frozen P4 baseline policies exist')
 OUT.mkdir(parents=True);atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','classification':'POSTHOC_EXPLORATORY_BASELINE','mode':MODE,'episodes_expected':1950,'scenario_source':'P5_NOMINAL_PREARMED_V2_CONTINUOUS / Round06 already-exposed final','selection_prohibited':True})
 res=[];rows=[]
 with ProcessPoolExecutor(max_workers=5) as ex:
  fs={ex.submit(lane,s):s for s in SEEDS}
  for f in as_completed(fs):
   s=fs[f]
   try:seed,rr,meta=f.result();rows.extend(rr);res.append(meta)
   except Exception as x:res.append({'seed':s,'state':'FAILED','error':str(x)})
   atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','classification':'POSTHOC_EXPLORATORY_BASELINE','completed':res,'episodes_expected':1950,'selection_prohibited':True})
 if len(rows)!=1950 or len(res)!=5: atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'FAILED','results':res,'episodes_written':len(rows)});raise SystemExit(1)
 with (OUT/'episodes.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'COMPLETE','classification':'POSTHOC_EXPLORATORY_BASELINE','results':res,'episodes_written':len(rows),'selection_prohibited':True})
if __name__=='__main__':main()

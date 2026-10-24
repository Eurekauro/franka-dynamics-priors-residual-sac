"""Five concurrent, one-seed lanes for the owner-approved PD+gravity baseline."""
import json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('PDG_OUTPUT_DIR',ROOT/'r3_mujoco/p4b_pd_gravity_attempt01'))
MODE='pd_gravity_residual';SEEDS=(201,202,203,204,205)
def atomic(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');os.replace(t,p)
def lane(seed):
 # p4b worker imports after the parent launcher exported P4B_OUTPUT_DIR.
 from p4b_full_campaign import worker
 return worker(MODE,seed,5)
def main():
 if OUT.exists():raise RuntimeError('append-only campaign output exists')
 OUT.mkdir(parents=True);atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','mode':MODE,'scheduler':'five_independent_seed_workers','parallelism':5,'seeds':SEEDS,'jobs_total':5,'transitions_per_seed':500000,'final_holdout_accessed':False,'started_unix_s':time.time()})
 t=time.perf_counter();res=[]
 with ProcessPoolExecutor(max_workers=5) as ex:
  fs={ex.submit(lane,s):s for s in SEEDS}
  for f in as_completed(fs):
   s=fs[f]
   try:r=f.result()
   except Exception as x:r={'mode':MODE,'seed':s,'state':'FAILED','error_type':type(x).__name__,'error':str(x)}
   res.append(r);atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','mode':MODE,'parallelism':5,'completed':res,'jobs_total':5,'final_holdout_accessed':False})
 state='COMPLETE' if len(res)==5 and all(x.get('state') in ('COMPLETE','ALREADY_COMPLETE') for x in res) else 'FAILED'
 atomic(OUT/'CAMPAIGN_STATUS.json',{'state':state,'mode':MODE,'parallelism':5,'results':res,'jobs_total':5,'wall_s':time.perf_counter()-t,'final_holdout_accessed':False})
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

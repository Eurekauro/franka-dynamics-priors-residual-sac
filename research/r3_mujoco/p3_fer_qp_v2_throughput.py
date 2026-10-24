"""Short real-control-loop process scaling benchmark; not a SAC experiment."""
import concurrent.futures as cf, csv, json, os, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'r3_mujoco/logs/p3_fer_qp_v2_throughput_attempt02'
MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual')
def worker(index,controls):
 os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 from p3_fer_qp_v2_env import PandaResidualFERQPV2Env
 # One representative, nonterminating prior is used for fair process scaling.
 mode='mlp_residual';e=PandaResidualFERQPV2Env(mode,'train',10000+index,max_controls=controls);t=time.perf_counter();e.reset(seed=100+index);startup=time.perf_counter()-t;t=time.perf_counter();n=0;reason=''
 while n<controls:
  _,_,term,trunc,info=e.step([0.]*7);n+=1
  if term or trunc:reason=info['termination_reason'];break
 return {'worker':index,'mode':mode,'startup_s':startup,'tracking_s':time.perf_counter()-t,'controls':n,'reason':reason}
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir(parents=True);rows=[];controls=128
 for workers in (1,2,3,4):
  t=time.perf_counter()
  with cf.ProcessPoolExecutor(max_workers=workers) as ex:r=list(ex.map(worker,range(workers),[controls]*workers))
  wall=time.perf_counter()-t; total=sum(x['controls'] for x in r);rows += [dict(x,processes=workers,combined_wall_s=wall,combined_controls=total,combined_controls_per_s=total/wall) for x in r]
 with (OUT/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 sums={str(n):{'controls_per_s':[z for z in rows if z['processes']==n][0]['combined_controls_per_s'],'combined_wall_s':[z for z in rows if z['processes']==n][0]['combined_wall_s'],'all_workers_completed':all(z['controls']==controls and not z['reason'] for z in rows if z['processes']==n)} for n in (1,2,3,4)}
 (OUT/'RUN_STATUS.json').write_text(json.dumps({'controls_per_worker':controls,'thread_policy':'OMP/OPENBLAS/MKL=1','results':sums},indent=2)+'\n')
 print(json.dumps(sums,indent=2))
if __name__=='__main__':main()

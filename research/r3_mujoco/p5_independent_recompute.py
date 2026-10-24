"""Read-only independent P5 aggregate verification from raw episode CSVs."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'r3_mujoco/p5_zero_shot_attempt01';OUT=RUN/'independent_recompute_v1';MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual')
def main():
 if OUT.exists():raise RuntimeError('append-only output exists')
 OUT.mkdir(); rows=[]
 for m in MODES:rows+=list(csv.DictReader((RUN/m/'episodes.csv').open()))
 original={(r['mode'],r['condition']):r for r in csv.DictReader((RUN/'SUMMARY_BY_MODE_CONDITION.csv').open())}
 maxdelta=0.;n=0
 for key,ref in original.items():
  x=[r for r in rows if (r['mode'],r['condition'])==key]; done=[r for r in x if not int(r['terminated'])]
  def rmse(a,k):
   steps=sum(int(z['steps']) for z in a);return np.sqrt(sum(float(z[k]) for z in a)/(steps*7)) if steps else np.nan
  vals={'completion_rate':len(done)/len(x),'mean_steps':np.mean([float(z['steps']) for z in x]),'q_rmse_completed_rad':rmse(done,'q_sse'),'dq_rmse_completed_rad_s':rmse(done,'dq_sse'),'q_rmse_prefix_rad':rmse(x,'q_sse'),'dq_rmse_prefix_rad_s':rmse(x,'dq_sse'),'mean_return':np.mean([float(z['return']) for z in x]),'projection_steps':sum(int(z['projection_steps']) for z in x)}
  for k,v in vals.items():
   if np.isnan(v) and ref[k]=='nan':continue
   maxdelta=max(maxdelta,abs(float(v)-float(ref[k])));n+=1
 if maxdelta>1e-12:raise RuntimeError(f'metric mismatch {maxdelta}')
 (OUT/'RECOMPUTE.json').write_text(json.dumps({'state':'PASS','raw_episode_rows':len(rows),'summary_cells_compared':n,'max_abs_delta':maxdelta,'final_holdout_accessed_by_recompute':False},indent=2)+'\n')
if __name__=='__main__':main()

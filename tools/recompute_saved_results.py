#!/usr/bin/env python3
"""Read-only arithmetic audit of migrated P4/P5 episode tables."""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def check(rows, qkey, dqkey):
 q=dq=0.
 for r in rows:
  n=int(float(r['steps'])); assert n>0
  q=max(q,abs(math.sqrt(float(r['q_sse'])/(7*n))-float(r[qkey])))
  dq=max(dq,abs(math.sqrt(float(r['dq_sse'])/(7*n))-float(r[dqkey])))
 return q,dq
def main():
 p4=read(ROOT/'results/p4b/five_arm_validation/episodes.csv')+read(ROOT/'results/p4b/pd_gravity_posthoc_validation/episodes.csv')
 strict=[]
 for p in sorted((ROOT/'results/p5_strict').glob('*/episodes.csv')): strict += read(p)
 posthoc=read(ROOT/'results/p5_pd_gravity_posthoc/episodes.csv')
 p4q,p4dq=check(p4,'q_rmse_rad','dq_rmse_rad_s')
 sq,sdq=check(strict,'q_rmse_prefix_rad','dq_rmse_prefix_rad_s')
 pq,pdq=check(posthoc,'q_rmse_prefix_rad','dq_rmse_prefix_rad_s')
 out={'state':'PASS','p4_rows':len(p4),'p5_strict_rows':len(strict),'p5_posthoc_rows':len(posthoc),
      'max_p4_abs_diff':[p4q,p4dq],'max_strict_p5_abs_diff':[sq,sdq],'max_posthoc_p5_abs_diff':[pq,pdq]}
 assert len(p4)==360 and len(strict)==9750 and len(posthoc)==1950
 assert max(p4q,p4dq,sq,sdq,pq,pdq)<1e-12
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

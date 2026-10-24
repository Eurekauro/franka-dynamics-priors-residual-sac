#!/usr/bin/env python3
"""Read-only recomputation of final P4/P5 evidence."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P4, P4G = ROOT/'p4b_full_attempt02', ROOT/'p4b_pd_gravity_attempt02'
P5, P5G = ROOT/'p5_nominal_prearmed_attempt06', ROOT/'p5_pd_gravity_exploratory_attempt01'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def fi(x): return int(float(x))
def ff(x): return float(x)

sources=[]
def src(label,p): sources.append({'label':label,'source_path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
def write(name,data):
    keys=sorted({x for r in data for x in r})
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)

OUT.mkdir(parents=True,exist_ok=True)
p4s=json.loads((P4/'CAMPAIGN_STATUS.json').read_text())
p4gs=json.loads((P4G/'CAMPAIGN_STATUS.json').read_text())
for n,p in [('P4B campaign',P4/'CAMPAIGN_STATUS.json'),('PDGravity campaign',P4G/'CAMPAIGN_STATUS.json'),('P5 strict campaign',P5/'CAMPAIGN_STATUS.json'),('P5 PDGravity campaign',P5G/'CAMPAIGN_STATUS.json')]: src(n,p)

policies=[]
def collect_policy(x, root, campaign):
    d=root/x['mode']/f"seed_{x['seed']}"
    rs,bc,model=d/'RUN_STATUS.json',d/'BEST_CHECKPOINT.json',d/'best_model.zip'
    for n,p in [('RUN_STATUS',rs),('BEST_CHECKPOINT',bc),('best_model',model)]: src(f'{campaign} {x["mode"]} seed{x["seed"]} {n}',p)
    policies.append({'campaign':campaign,'mode':x['mode'],'seed':x['seed'],'state':x['state'],'timesteps':x['timesteps'],'policy_path':str(model),'policy_sha256':digest(model),'run_status_sha256':digest(rs),'best_checkpoint_sha256':digest(bc)})
for group in p4s['results']:
    for x in group['seed_results']: collect_policy(x,P4,'P4B')
for x in p4gs['results']: collect_policy(x,P4G,'PD_GRAVITY_EXTENSION')

validation=[]
def collect_validation(path,label):
    src(label,path)
    for r in rows(path):
        st=fi(r['steps']); q=math.sqrt(ff(r['q_sse'])/(7*st)); dq=math.sqrt(ff(r['dq_sse'])/(7*st))
        validation.append({'mode':r['mode'],'policy_seed':fi(r['policy_seed']),'validation_seed':fi(r['validation_seed']),'task':r['task'],'steps':st,'terminated':fi(r['terminated']),'reason':r['termination_reason'] or 'NONE','q_rmse_stored':ff(r['q_rmse_rad']),'q_rmse_recomputed':q,'q_rmse_abs_diff':abs(q-ff(r['q_rmse_rad'])),'dq_rmse_stored':ff(r['dq_rmse_rad_s']),'dq_rmse_recomputed':dq,'dq_rmse_abs_diff':abs(dq-ff(r['dq_rmse_rad_s']))})
collect_validation(P4/'independent_selected_validation_v1/episodes.csv','P4B validation')
collect_validation(P4G/'independent_selected_validation_v1/episodes.csv','PDGravity validation')

def collect_p5(base,label,classification,onefile=False):
    ans=[]
    files=[base/'episodes.csv'] if onefile else sorted(base.glob('*/episodes.csv'))
    for p in files:
        src(f'{label} {p.parent.name} episodes',p)
        for r in rows(p):
            st=fi(r['steps']); q=math.sqrt(ff(r['q_sse'])/(7*st)); dq=math.sqrt(ff(r['dq_sse'])/(7*st))
            ans.append({'mode':r['mode'],'policy_seed':fi(r['policy_seed']),'condition':r['condition'],'scenario_seed':fi(r['scenario_seed']),'steps':st,'terminated':fi(r['terminated']),'reason':r['termination_reason'] or 'NONE','projection_steps':fi(r['projection_steps']),'q_rmse_prefix_rad':ff(r['q_rmse_prefix_rad']),'dq_rmse_prefix_rad_s':ff(r['dq_rmse_prefix_rad_s']),'q_rmse_abs_diff':abs(q-ff(r['q_rmse_prefix_rad'])),'dq_rmse_abs_diff':abs(dq-ff(r['dq_rmse_prefix_rad_s'])),'classification':classification})
    return ans
p5strict=collect_p5(P5,'P5 strict','STRICT_NOMINAL_ONLY_CONFIRMATORY')
p5grav=collect_p5(P5G,'P5 PDGravity','POSTHOC_EXPLORATORY_BASELINE',True)

def summary(data,keys,dataset):
    group=defaultdict(list)
    for r in data: group[tuple(r[k] for k in keys)].append(r)
    ans=[]
    for k,rs in sorted(group.items()):
        steps=sum(r['steps'] for r in rs); complete=sum(r['steps']==500 and not r['terminated'] for r in rs)
        qkey='q_rmse_prefix_rad' if 'q_rmse_prefix_rad' in rs[0] else 'q_rmse_recomputed'
        dqkey='dq_rmse_prefix_rad_s' if 'dq_rmse_prefix_rad_s' in rs[0] else 'dq_rmse_recomputed'
        d=dict(zip(keys,k)); d.update({'dataset':dataset,'episodes':len(rs),'completed':complete,'completion_rate':complete/len(rs),'mean_steps':steps/len(rs),'mean_q_rmse_prefix':sum(r[qkey] for r in rs)/len(rs),'mean_dq_rmse_prefix':sum(r[dqkey] for r in rs)/len(rs),'projection_step_rate':sum(r.get('projection_steps',0) for r in rs)/steps,'termination_reasons':json.dumps(dict(Counter(r['reason'] for r in rs)),sort_keys=True)})
        ans.append(d)
    return ans

write('SOURCE_HASHES.csv',sources);write('POLICY_REGISTRY_AUDITED.csv',policies);write('P4_VALIDATION_RECOMPUTED.csv',validation);write('P4_VALIDATION_MODE_SUMMARY.csv',summary(validation,['mode'],'FROZEN_VALIDATION'));write('P5_STRICT_EPISODE_RECOMPUTE.csv',p5strict);write('P5_STRICT_MODE_CONDITION_SUMMARY.csv',summary(p5strict,['mode','condition'],'STRICT_NOMINAL_ONLY_CONFIRMATORY'));write('P5_STRICT_MODE_SUMMARY.csv',summary(p5strict,['mode'],'STRICT_NOMINAL_ONLY_CONFIRMATORY'));write('P5_PD_GRAVITY_EXPLORATORY_EPISODE_RECOMPUTE.csv',p5grav);write('P5_PD_GRAVITY_EXPLORATORY_SUMMARY.csv',summary(p5grav,['mode','condition'],'POSTHOC_EXPLORATORY_BASELINE'))
status={'status':'PASS_WITH_LIMITATIONS','policies':len(policies),'transitions':sum(x['timesteps'] for x in policies),'p4_validation_rows':len(validation),'p5_strict_rows':len(p5strict),'p5_gravity_rows':len(p5grav),'max_validation_q_diff':max(x['q_rmse_abs_diff'] for x in validation),'max_validation_dq_diff':max(x['dq_rmse_abs_diff'] for x in validation),'max_p5_strict_q_diff':max(x['q_rmse_abs_diff'] for x in p5strict),'max_p5_strict_dq_diff':max(x['dq_rmse_abs_diff'] for x in p5strict),'max_p5_gravity_q_diff':max(x['q_rmse_abs_diff'] for x in p5grav),'max_p5_gravity_dq_diff':max(x['dq_rmse_abs_diff'] for x in p5grav)}
(OUT/'AUDIT_STATUS.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
(OUT/'AUDIT_REPORT.md').write_text('''# Final Consolidation Independent Audit\n\nThis is a read-only audit of saved campaign artifacts.\n\n- P4-B status records 25 policies (five modes x five seeds), each at 500,000 transitions.\n- PD+Gravity extension records five additional 500,000-transition policies. The audited total is therefore 30 policies / 15.0M transitions.\n- Strict P5 Round06 supplies 9,750 episodes (five original modes x five seeds x 13 conditions x 30 new scenarios) under the repaired nominal-only controller.\n- PD+Gravity P5 supplies 1,950 episodes but is `POSTHOC_EXPLORATORY_BASELINE`: it reused exposed Round06 scenarios and is not a sixth confirmatory arm.\n\nRMSE is independently recomputed from stored SSE and executed steps. Completion requires exactly 500 steps and no termination. Prefix RMSE remains separate from completed-episode tracking.\n\n## Limitations\n\nThis checks stored artifacts/arithmetic, not retraining. Training seeds are the primary repetitions; validation scenarios have documented reference repetition. Strict P5 is simulation research, not hardware/FER certification. Earlier mismatch-aware/startup-defective P5 outputs are not aggregated here.\n''')
print(json.dumps(status,indent=2))

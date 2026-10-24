"""Frozen P4-B 25-job SAC campaign under FER-QP V2.1."""
import csv,json,os,random,resource,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('P4B_OUTPUT_DIR',str(ROOT/'r3_mujoco/p4b_full_attempt01')));MODES=('pure_sac','pd_residual','mlp_residual','gp_residual','rbf_residual');SEEDS=(201,202,203,204,205)
def atomic(p,o):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2)+'\n');os.replace(t,p)
def p4():return json.loads((ROOT/'r3_mujoco/P3_P4_P5_FER_QP_V2_PROTOCOL.json').read_text())
def add():return json.loads((ROOT/'r3_mujoco/P4B_FULL_TRAINING_ADDENDUM_V1.json').read_text())
def snapshot(path):
 import torch
 name,keys,pos,has_gauss,cached=np.random.get_state();path.with_suffix('.json').write_text(json.dumps({'numpy_algorithm':name,'numpy_position':int(pos),'numpy_has_gauss':int(has_gauss),'numpy_cached_gaussian':float(cached),'python_state':repr(random.getstate())},sort_keys=True)+'\n');np.savez_compressed(path,numpy_keys=np.asarray(keys,dtype=np.uint32),torch_cpu=torch.get_rng_state().numpy())
def restore(path):
 import ast,torch
 meta=json.loads(path.with_suffix('.json').read_text());z=np.load(path,allow_pickle=False);np.random.set_state((meta['numpy_algorithm'],np.asarray(z['numpy_keys'],dtype=np.uint32),int(meta['numpy_position']),int(meta['numpy_has_gauss']),float(meta['numpy_cached_gaussian'])));random.setstate(ast.literal_eval(meta['python_state']));torch.set_rng_state(torch.from_numpy(np.asarray(z['torch_cpu'],dtype=np.uint8)))
def evaluate(model,mode,p,out,timestep):
 from p3_fer_qp_v2_env import PandaResidualFERQPV2Env
 rows=[]
 for seed in p['scenarios']['validation_seeds']:
  env=PandaResidualFERQPV2Env(mode,'validation',int(seed));obs,_=env.reset(seed=int(seed));ret=qse=0.;steps=proj=0;term_reason=''
  while True:
   a,_=model.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(a);qr,_,_=env._reference();q=env.d.qpos[env.s.qadr];qse+=float((q-qr)@(q-qr));ret+=float(r);steps+=1;proj+=int(info['projected'])
   if term or trunc:term_reason=info['termination_reason'];break
  rows.append({'checkpoint_timesteps':timestep,'validation_seed':seed,'return':ret,'steps':steps,'terminated':int(bool(term_reason)),'termination_reason':term_reason,'projection_steps':proj,'q_sse':qse,'q_rmse_rad':float(np.sqrt(qse/(steps*7)))})
 f=out/'validation.csv';old=list(csv.DictReader(f.open())) if f.exists() else []
 with f.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(old+rows)
def choose_checkpoint(out):
 rows=list(csv.DictReader((out/'validation.csv').open()));choices=[]
 for t in sorted({int(x['checkpoint_timesteps']) for x in rows}):
  r=[x for x in rows if int(x['checkpoint_timesteps'])==t];complete=[x for x in r if x['terminated']=='0'];choices.append({'checkpoint_timesteps':t,'complete_episodes':len(complete),'mean_steps':float(np.mean([float(x['steps']) for x in r])),'q_rmse_completed':float(np.mean([float(x['q_rmse_rad']) for x in complete])) if complete else float('inf'),'mean_return':float(np.mean([float(x['return']) for x in r]))})
 # Ordered criterion, then earliest timestep.  Inf is deliberately worst.
 best=sorted(choices,key=lambda x:(-x['complete_episodes'],-x['mean_steps'],x['q_rmse_completed'],-x['mean_return'],x['checkpoint_timesteps']))[0]
 atomic(out/'BEST_CHECKPOINT.json',{'rule':add()['best_checkpoint_rule'],'candidates':choices,'selected':best})
 return best
def worker(mode,seed,parallelism):
 os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1';import torch
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 from stable_baselines3 import SAC
 from stable_baselines3.common.callbacks import BaseCallback
 from p3_fer_qp_v2_env import PandaResidualFERQPV2Env
 p=p4();a=add();d=OUT/mode/f'seed_{seed:03d}';d.mkdir(parents=True,exist_ok=True)
 if (d/'final_model.zip').exists():return {'mode':mode,'seed':seed,'state':'ALREADY_COMPLETE'}
 class CB(BaseCallback):
  def __init__(self,base):super().__init__();self.next=((base//a['validation']['every_transitions'])+1)*a['validation']['every_transitions'];self.start=time.perf_counter()
  def _on_step(self):
   if time.perf_counter()-self.start>a['resources']['wall_hours_max']*3600:raise RuntimeError('per-job wall resource gate')
   if self.num_timesteps>=self.next:
    evaluate(self.model,mode,p,d,self.num_timesteps);self.model.save(d/f'checkpoint_{self.num_timesteps:06d}');self.model.save_replay_buffer(d/f'replay_{self.num_timesteps:06d}.pkl');snapshot(d/f'rng_{self.num_timesteps:06d}.npz');self.next+=a['validation']['every_transitions']
   return True
 atomic(d/'RUN_STATUS.json',{'state':'RUNNING','mode':mode,'seed':seed,'pid':os.getpid(),'parallelism':parallelism})
 try:
  np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);env=PandaResidualFERQPV2Env(mode,'train');c=p['sac'];ck=sorted(d.glob('checkpoint_*.zip'));base=0
  if ck:
   last=ck[-1];m=SAC.load(last,env=env,device='cpu');base=int(m.num_timesteps);replay=d/(last.stem.replace('checkpoint_','replay_')+'.pkl');rng=d/(last.stem.replace('checkpoint_','rng_')+'.npz')
   if not replay.exists() or not rng.exists() or not rng.with_suffix('.json').exists():raise RuntimeError('incomplete checkpoint recovery set')
   m.load_replay_buffer(replay);restore(rng)
  else:m=SAC(c['policy'],env,learning_rate=c['learning_rate'],buffer_size=c['buffer_size'],learning_starts=c['learning_starts'],batch_size=c['batch_size'],gamma=c['gamma'],tau=c['tau'],train_freq=c['train_freq'],gradient_steps=c['gradient_steps'],ent_coef=c['ent_coef'],policy_kwargs={'net_arch':c['net_arch']},seed=seed,verbose=0,device='cpu')
  remaining=a['transitions_per_seed']-base
  if remaining<=0:raise RuntimeError('invalid recovered timestep count')
  t=time.perf_counter();m.learn(total_timesteps=remaining,callback=CB(base),progress_bar=False,reset_num_timesteps=False);wall=time.perf_counter()-t;m.save(d/'final_model');m.save_replay_buffer(d/'final_replay.pkl');snapshot(d/'final_rng.npz');best=choose_checkpoint(d)
  ck=d/f"checkpoint_{best['checkpoint_timesteps']:06d}.zip";replay=d/f"replay_{best['checkpoint_timesteps']:06d}.pkl";rng=d/f"rng_{best['checkpoint_timesteps']:06d}.npz";import shutil;shutil.copy2(ck,d/'best_model.zip');shutil.copy2(replay,d/'best_replay.pkl');shutil.copy2(rng,d/'best_rng.npz');shutil.copy2(rng.with_suffix('.json'),d/'best_rng.json')
  r={'mode':mode,'seed':seed,'state':'COMPLETE','timesteps':a['transitions_per_seed'],'resumed_from_timesteps':base,'wall_s':wall,'cpu_s':time.process_time(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'selected_checkpoint':best,'device':str(m.device)};atomic(d/'RUN_STATUS.json',r);return r
 except Exception as e:
  r={'mode':mode,'seed':seed,'state':'FAILED','error_type':type(e).__name__,'error':str(e)};atomic(d/'RUN_STATUS.json',r);raise
def main():
 a=add();resuming=OUT.exists();OUT.mkdir(parents=True,exist_ok=True);n=int(os.environ.get('P4B_PARALLELISM',a['resources']['initial_parallel_workers']))
 if n not in (1,2,3,4):raise ValueError('parallelism must be 1..4')
 jobs=[(m,s) for m in MODES for s in SEEDS]
 if resuming:
  with (OUT/'RECOVERY_EVENTS.jsonl').open('a') as f:f.write(json.dumps({'event':'checkpoint_boundary_resume','time_unix_s':time.time(),'parallelism':n})+'\n')
 atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','parallelism':n,'jobs_total':len(jobs),'protocol':'P4B_FULL_TRAINING_ADDENDUM_V1','final_holdout_accessed':False,'resuming':resuming});start=time.perf_counter();results=[]
 with ProcessPoolExecutor(max_workers=n) as ex:
  fs={ex.submit(worker,m,s,n):(m,s) for m,s in jobs}
  for f in as_completed(fs):
   m,s=fs[f]
   try:results.append(f.result())
   except Exception as e:results.append({'mode':m,'seed':s,'state':'FAILED','error':str(e)})
   atomic(OUT/'CAMPAIGN_STATUS.json',{'state':'RUNNING','parallelism':n,'completed':results,'jobs_total':len(jobs),'final_holdout_accessed':False})
 total_cpu=sum(x.get('cpu_s',0.) for x in results);state='COMPLETE' if len(results)==len(jobs) and all(x['state'] in ('COMPLETE','ALREADY_COMPLETE') for x in results) else 'FAILED'
 atomic(OUT/'CAMPAIGN_STATUS.json',{'state':state,'parallelism':n,'results':results,'jobs_total':len(jobs),'wall_s':time.perf_counter()-start,'cpu_core_hours':total_cpu/3600,'final_holdout_accessed':False})
 if state!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

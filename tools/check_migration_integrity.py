#!/usr/bin/env python3
"""Verify copied source assets and core runtime assets from the migration ledger."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/yuli/projects/franka-panda-lip-residual-rl')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 rows=list(csv.DictReader((ROOT/'internal_archive/provenance/MIGRATION_MANIFEST.csv').open()))
 failed=[]
 for r in rows:
  p=Path(r['source_path'])
  if not p.exists() or sha(p)!=r['source_sha256']: failed.append(str(p))
 # All unmodified copied source evidence lives in source_snapshot. Refactored
 # runtime files are verified by the independent executable smoke test.
 snapshot=ROOT/'internal_archive/source_snapshot'
 missing=[]
 for r in rows:
  rel=Path(r['source_path']).relative_to(SOURCE)
  q=snapshot/rel
  if not q.exists() or sha(q)!=r['source_sha256']: missing.append(str(rel))
 out={'manifest_rows':len(rows),'source_hash_mismatches':failed,'snapshot_hash_mismatches':missing,
      'state':'PASS' if not failed and not missing else 'FAIL'}
 print(json.dumps(out,indent=2))
 if failed or missing: raise SystemExit(1)
if __name__=='__main__':main()

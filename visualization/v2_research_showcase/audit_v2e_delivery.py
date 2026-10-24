#!/usr/bin/env python3
"""Static V2-E delivery inventory; never opens a writer for media files."""
from __future__ import annotations
import hashlib,json,mimetypes
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent/'outputs_v2e'/'attempt_01'
FINAL=['v2e_github_highlight_reel_final.mp4','v2e_research_demonstration_final.mp4','v2e_readme_formal_motion.gif','v2e_cover.png']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def meta(p):
 x={'relative_path':p.name,'sha256':sha(p),'size_bytes':p.stat().st_size,'mime_type':mimetypes.guess_type(p.name)[0]}
 if p.suffix=='.mp4':
  r=imageio.get_reader(p);m=r.get_meta_data();r.get_data(0);r.close();x|={'media_type':'MP4','duration_s':float(m['duration']),'resolution_px':list(m['size']),'fps':float(m['fps']),'codec':m.get('codec'),'pixel_format':m.get('pix_fmt')}
 elif p.suffix=='.gif':
  im=Image.open(p);x|={'media_type':'GIF','resolution_px':list(im.size),'frame_count':int(getattr(im,'n_frames',1))}
 elif p.suffix=='.png':
  im=Image.open(p);x|={'media_type':'PNG','resolution_px':list(im.size)}
 return x
sources=[
 ('P5 replay: nominal multi-joint seed 40019','results/final_consolidation/visualization/high_quality_demos_attempt02/four_controller_sync_multi_joint_seed40019.mp4','P5_PROTOCOL_MATCHED_VISUALIZATION_REPLAY'),
 ('P5 replay: nominal point-to-point seed 40023','results/final_consolidation/visualization/high_quality_demos_attempt02/four_controller_sync_point_to_point_seed40023.mp4','P5_PROTOCOL_MATCHED_VISUALIZATION_REPLAY'),
 ('P5 replay: nominal single-joint seed 40015','results/final_consolidation/visualization/high_quality_demos_attempt02/four_controller_sync_single_joint_auxiliary_seed40015.mp4','P5_PROTOCOL_MATCHED_VISUALIZATION_REPLAY'),
 ('P5 replay: RBF seed 201 nominal vs combined_strong','results/final_consolidation/visualization/high_quality_demos_attempt02/nominal_vs_combined_strong_rbf_seed40023.mp4','P5_PROTOCOL_MATCHED_VISUALIZATION_REPLAY'),
 ('P4 validation six-group summary','docs/assets/figures/p4b_validation_six_group_summary.png','P4_FROZEN_VALIDATION_RESULT'),
 ('P4 validation learning curves','docs/assets/figures/p4b_validation_learning_curves.png','P4_FROZEN_VALIDATION_LEARNING_CURVE'),
 ('P5 strict completion heatmap','docs/assets/figures/p5_strict_completion_heatmap.png','P5_CONFIRMATORY_FROZEN_RESULT'),
 ('P5 termination reasons','docs/assets/figures/p5_termination_reasons.png','P5_CONFIRMATORY_FROZEN_RESULT'),
 ('P5 strict vs posthoc completion','docs/assets/figures/p5_overall_completion_strict_vs_posthoc.png','P5_CONFIRMATORY_AND_POSTHOC_BOUNDARY')]
manifest={'schema_version':'2.0','classification':'P5 Protocol-Matched Visualization Replay / frozen-result presentation','final_delivery_files':[meta(OUT/n) for n in FINAL],'other_local_delivery_files':[meta(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in FINAL and p.name!='MANIFEST.json'],'source_materials':[{'label':l,'relative_path':path,'evidence_attribute':tag,'accessible':(ROOT/path).exists(),'sha256':sha(ROOT/path) if (ROOT/path).exists() else None} for l,path,tag in sources],'reproducibility_status':{'media_integrity':'VERIFIABLE','source_provenance':'VERIFIABLE_FOR_LISTED_FILES','run_v2e_final_py':'Produces non-final base names only; does not itself reproduce final _final.mp4 files.','historical_final_assembly':'VERIFIED_FROM_RETAINED_LOCAL_COMMAND_RECORD: base V2-E MP4s were extended with 0.5x single-joint replay/P4 summary (highlight) and three frozen figures (research demo). No standalone historical assembly script was retained.','complete_final_rebuild':'UNAVAILABLE_FROM_A_SINGLE_RETAINED_SCRIPT','p4_motion':'NO_ARCHIVED_PER_STATE_TRAJECTORY_FOUND; frozen P4 figures only','p4_p5_scientific_reproduction':'NOT_CLAIMED; frozen models/full artifacts are not part of this delivery audit'},'media_mutation':'No final media bytes were opened for writing by this audit.'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

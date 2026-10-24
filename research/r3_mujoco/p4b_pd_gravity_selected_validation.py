"""Independent Validation replay of the sixth baseline only; never opens Final."""
from pathlib import Path
import os
import p4b_selected_validation as v
ROOT=Path(__file__).resolve().parents[1]
v.CAMPAIGN=Path(os.environ.get('PDG_P4_OUTPUT_DIR',ROOT/'r3_mujoco/p4b_pd_gravity_attempt02'))
v.OUT=v.CAMPAIGN/'independent_selected_validation_v1'
v.MODES=('pd_gravity_residual',)
if __name__=='__main__':v.main()

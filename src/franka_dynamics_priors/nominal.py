"""Frozen nominal PD and analytical gravity helpers used by the final protocol."""
# SPDX-License-Identifier: Apache-2.0
import mujoco
import numpy as np

# Preserved from the P0/P2 handoff controller.  They are not SAC parameters.
KP = np.array([120., 120., 100., 100., 45., 45., 35.])
KD = 1.4 * np.sqrt(KP)

def gravity(model: mujoco.MjModel, data: mujoco.MjData, safety) -> np.ndarray:
    """Return nominal gravity/bias with velocity temporarily zeroed, exactly as P2."""
    velocity = data.qvel.copy()
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    torque = data.qfrc_bias[safety.dof] - data.qfrc_passive[safety.dof]
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    return torque

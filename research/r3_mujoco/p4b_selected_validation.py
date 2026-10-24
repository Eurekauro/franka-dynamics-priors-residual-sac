"""Independent, frozen-validation replay for all selected P4-B policies.

This intentionally never instantiates a Final Holdout environment.  It writes
only numeric error arrays (NPZ, no pickle) and an episode ledger, so metrics can
be recomputed outside SB3.
"""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from p3_fer_qp_v2_env import PandaResidualFERQPV2Env, protocol

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path(os.environ.get("P4B_OUTPUT_DIR", ROOT / "r3_mujoco/p4b_full_attempt02"))
OUT = CAMPAIGN / "independent_selected_validation_v1"
MODES = ("pure_sac", "pd_residual", "mlp_residual", "gp_residual", "rbf_residual")
SEEDS = (201, 202, 203, 204, 205)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError(f"append-only output exists: {OUT}")
    p = protocol()
    validation = tuple(map(int, p["scenarios"]["validation_seeds"]))
    if len(validation) != 12:
        raise ValueError("frozen protocol must contain exactly 12 validation seeds")
    OUT.mkdir(parents=True)
    rows, arrays, inventory = [], {}, []
    start = time.perf_counter()
    for mode in MODES:
        for policy_seed in SEEDS:
            base = CAMPAIGN / mode / f"seed_{policy_seed:03d}"
            status = json.loads((base / "RUN_STATUS.json").read_text())
            choice = json.loads((base / "BEST_CHECKPOINT.json").read_text())
            if status.get("state") != "COMPLETE":
                raise RuntimeError(f"incomplete policy {mode}/{policy_seed}")
            model_path = base / "best_model.zip"
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            model_hash = sha256(model_path)
            inventory.append({"mode": mode, "policy_seed": policy_seed,
                              "best_model_sha256": model_hash,
                              "selected_checkpoint": choice["selected"]["checkpoint_timesteps"]})
            model = SAC.load(model_path, device="cpu")
            for scenario_seed in validation:
                env = PandaResidualFERQPV2Env(mode, "validation", scenario_seed)
                obs, _ = env.reset(seed=scenario_seed)
                ret = qse = dqse = tau2 = projection2 = 0.0
                qhist, dqhist, rawhist, safehist = [], [], [], []
                projected = 0
                reason = ""
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    q = np.asarray(env.d.qpos[env.s.qadr], dtype=np.float64)
                    dq = np.asarray(env.d.qvel[env.s.dof], dtype=np.float64)
                    qref, dqref, _ = env._reference()
                    raw = np.asarray(info["raw_torque_nm"], dtype=np.float64)
                    safe = np.asarray(info["safe_torque_nm"], dtype=np.float64)
                    if not np.isfinite(np.r_[q, dq, qref, dqref, raw, safe]).all():
                        raise FloatingPointError(f"nonfinite {mode}/{policy_seed}/{scenario_seed}")
                    qe, dqe = q - qref, dq - dqref
                    qse += float(qe @ qe)
                    dqse += float(dqe @ dqe)
                    tau2 += float(safe @ safe)
                    projection2 += float((raw - safe) @ (raw - safe))
                    qhist.append(qe); dqhist.append(dqe); rawhist.append(raw); safehist.append(safe)
                    ret += float(reward); projected += int(info["projected"])
                    if terminated or truncated:
                        reason = str(info.get("termination_reason", ""))
                        break
                n = len(qhist)
                key = f"{mode}__p{policy_seed}__v{scenario_seed}"
                arrays[key + "__q_error"] = np.asarray(qhist, dtype=np.float64)
                arrays[key + "__dq_error"] = np.asarray(dqhist, dtype=np.float64)
                arrays[key + "__raw_torque"] = np.asarray(rawhist, dtype=np.float64)
                arrays[key + "__safe_torque"] = np.asarray(safehist, dtype=np.float64)
                rows.append({"mode": mode, "policy_seed": policy_seed,
                             "validation_seed": scenario_seed,
                             "task": info["scenario"]["task"], "steps": n,
                             "return": ret, "terminated": int(bool(reason)),
                             "termination_reason": reason, "projection_steps": projected,
                             "q_sse": qse, "dq_sse": dqse, "safe_tau_sse": tau2,
                             "raw_minus_safe_sse": projection2,
                             "q_rmse_rad": float(np.sqrt(qse / (n * 7))),
                             "dq_rmse_rad_s": float(np.sqrt(dqse / (n * 7))),
                             "best_model_sha256": model_hash})
    with (OUT / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / "model_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)
    np.savez_compressed(OUT / "errors_and_torques.npz", **arrays)
    (OUT / "RUN_STATUS.json").write_text(json.dumps({
        "state": "COMPLETE", "episodes": len(rows), "final_holdout_accessed": False,
        "validation_seeds": validation, "wall_s": time.perf_counter() - start,
        "numeric_evidence": "errors_and_torques.npz; require allow_pickle=False"
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

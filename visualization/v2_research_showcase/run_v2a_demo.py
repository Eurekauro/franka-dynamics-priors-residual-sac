#!/usr/bin/env python3
"""Create V2-A DEMO_ONLY feasibility records and a real MuJoCo preview.

This runner is deliberately outside the tracked research package.  It does not
alter any P4/P5 configuration, model, policy, safety bound, or metric.  All
commands pass through PandaResidualFERQPV2Env and its frozen FER-QP V2.1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import types
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs_v2a_run04"
DT = 0.004
FPS = 30


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()


@dataclass
class Reference:
    name: str
    q: np.ndarray
    dt: float = DT

    def __post_init__(self) -> None:
        order = 2 if len(self.q) >= 3 else 1
        self.v = np.gradient(self.q, self.dt, axis=0, edge_order=order)
        self.acc = np.gradient(self.v, self.dt, axis=0, edge_order=order)

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.clip(t / self.dt, 0.0, len(self.q) - 1.0)
        i = int(np.floor(u))
        a = u - i
        j = min(i + 1, len(self.q) - 1)
        q = (1 - a) * self.q[i] + a * self.q[j]
        v = (1 - a) * self.v[i] + a * self.v[j]
        acc = (1 - a) * self.acc[i] + a * self.acc[j]
        return q, v, acc

    @property
    def seconds(self) -> float:
        return (len(self.q) - 1) * self.dt


def smoothstep5(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def sweep(home: np.ndarray, dt: float = DT) -> Reference:
    # Home -> left -> right -> forward/up -> wrist emphasis -> home.  These
    # offsets retain substantial margin to all frozen Panda joint limits.
    offsets = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.55, -0.36, 0.38, -0.30, 0.30, -0.26, 0.28],
        [-0.58, 0.34, -0.42, 0.32, -0.34, 0.34, -0.30],
        [0.18, -0.54, 0.46, -0.50, 0.42, 0.20, 0.34],
        [-0.20, 0.08, 0.14, 0.10, -0.52, 0.46, -0.48],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    segment = 1.9
    n = int(round(segment / dt))
    chunks = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        u = smoothstep5(np.linspace(0, 1, n, endpoint=False))[:, None]
        chunks.append(home + (1 - u) * a + u * b)
    return Reference("A_multi_pose_sweeping", np.vstack(chunks + [home[None]]))


def excitation(home: np.ndarray, dt: float = DT) -> Reference:
    seconds = 10.0
    t = np.arange(0, seconds + dt / 2, dt)
    amp = np.array([0.36, 0.30, 0.38, 0.32, 0.30, 0.34, 0.28])
    hz = np.array([0.11, 0.15, 0.13, 0.17, 0.09, 0.14, 0.12])
    phase = np.array([0.0, 0.8, 1.7, 2.2, 0.5, 2.8, 1.2])
    # A 1 second quintic ramp keeps the startup reference differentiable.
    ramp = smoothstep5(t / 1.0)
    q = home + ramp[:, None] * amp * np.sin(2 * np.pi * hz * t[:, None] + phase)
    return Reference("C_multi_joint_excitation", q)


def ee_position(model: mujoco.MjModel, data: mujoco.MjData, qadr: np.ndarray, q: np.ndarray, body_id: int) -> np.ndarray:
    data.qpos[qadr] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data.xpos[body_id].copy()


def figure_eight(model: mujoco.MjModel, qadr: np.ndarray, qmin: np.ndarray, qmax: np.ndarray, home: np.ndarray, dt: float = DT) -> Reference:
    """Damped-least-squares Cartesian figure-eight, generated from Panda FK."""
    data = mujoco.MjData(model)
    hand = model.body("hand").id
    center = ee_position(model, data, qadr, home, hand)
    duration = 9.0
    t = np.arange(0, duration + dt / 2, dt)
    # One horizontal figure-eight at a modest depth/height excursion.  This
    # is a kinematic DEMO_ONLY reference, not a task learned by SAC.
    theta = 2 * np.pi * t / 6.0
    target = center + np.c_[0.11 * np.sin(theta), 0.075 * np.sin(2 * theta), 0.045 * (np.cos(theta) - 1)]
    q = home.copy()
    qs = []
    for goal in target:
        pos = ee_position(model, data, qadr, q, hand)
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand)
        j = jacp[:, model.jnt_dofadr[[model.joint(f"joint{i}").id for i in range(1, 8)]]]
        # Low-gain resolved-rate IK makes consecutive reference samples
        # smooth and avoids a fabricated inverse-kinematics jump.
        dq = j.T @ np.linalg.solve(j @ j.T + 0.02 * np.eye(3), 7.0 * (goal - pos))
        q = np.clip(q + dt * dq, qmin + 0.08, qmax - 0.08)
        qs.append(q.copy())
    return Reference("B_3d_figure_eight", np.asarray(qs))


def install_reference(env, ref: Reference) -> None:
    def get_reference(self):
        return ref.at(self.k * DT)
    env._reference = types.MethodType(get_reference, env)


def make_env(mode: str, ref: Reference):
    from franka_dynamics_priors.env import PandaResidualFERQPV2Env
    # A valid task identity preserves the frozen 39-D observation layout; the
    # actual q/dq/ddq references are replaced only in this DEMO_ONLY runner.
    e = PandaResidualFERQPV2Env(mode, "train", 10000, max_controls=len(ref.q) - 1,
                                nominal_control_isolation=True,
                                scenario_definition={"seed": -1, "split": "demo_only", "task": "multi_joint", "amplitude_multiplier": 1.0, "frequency_hz": 0.0, "phase_offset_rad": 0.0, "single_joint_index": None})
    e.reset(seed=10000)
    install_reference(e, ref)
    return e


def fk(env, q: np.ndarray) -> np.ndarray:
    model, data = env.m, env.d_control if env.d_control is not None else env.d
    saved = data.qpos.copy(), data.qvel.copy()
    p = ee_position(model, data, env.s.qadr, q, model.body("hand").id)
    data.qpos[:], data.qvel[:] = saved
    mujoco.mj_forward(model, data)
    return p


def rollout(label: str, mode: str, ref: Reference, policy: SAC | None, render: bool = False, style: str = "light") -> tuple[dict, list[dict]]:
    env = make_env(mode, ref)
    records: list[dict] = []
    start = time.perf_counter()
    frame_rows: list[dict] = []
    renderer = None
    writer = None
    video_path = None
    if render:
        video_path = OUT / "v2a_motion_and_visual_style_preview.mp4"
        # The tracked MJCF declares a 640x480 offscreen target.  Enlarging
        # this renderer-only buffer after model load changes neither physics
        # parameters nor any formal controller/safety setting.
        env.m.vis.global_.offwidth = 1920
        env.m.vis.global_.offheight = 1080
        renderer = mujoco.Renderer(env.m, height=1080, width=1920)
        camera = mujoco.MjvCamera(); camera.azimuth = 138; camera.elevation = -18; camera.distance = 1.55; camera.lookat = np.array([0.0, 0.0, 0.52])
        writer = imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        next_frame = 0.0
    try:
        obs = env._obs(*env._reference())
        termination = ""
        for k in range(len(ref.q) - 1):
            action = np.zeros(7, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, term, trunc, info = env.step(action)
            qr, _, _ = env._reference()
            q = env.d.qpos[env.s.qadr].copy()
            actual = fk(env, q); reference = fk(env, qr)
            row = {"step": k + 1, "sim_time_s": (k + 1) * DT, "q": q.tolist(), "qref": qr.tolist(), "ee_actual_m": actual.tolist(), "ee_reference_m": reference.tolist(), "tracking_error_rad": float(np.sqrt(np.mean((q - qr) ** 2))), "projected": bool(info["projected"]), "projection_norm_nm": float(info["qp"].get("projection_norm_nm", 0.0)), "termination_reason": info["termination_reason"]}
            records.append(row)
            if render and row["sim_time_s"] + 1e-10 >= next_frame:
                renderer.update_scene(env.d, camera=camera)
                frame_rows.append(row)
                writer.append_data(overlay(renderer.render(), frame_rows, label, style))
                next_frame += 1.0 / FPS
            if term or trunc:
                termination = info["termination_reason"] or ("horizon_complete" if trunc else "terminated")
                break
        else:
            termination = "reference_exhausted"
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
        env.close()
    q_array = np.asarray([r["q"] for r in records]); ee = np.asarray([r["ee_actual_m"] for r in records])
    ref_ee = np.asarray([r["ee_reference_m"] for r in records])
    summary = {"label": label, "mode": mode, "controller_identity": "PD + analytical gravity, zero SAC residual" if policy is None else "frozen SAC policy seed 201", "reference": ref.name, "demo_classification": "DEMO_ONLY", "steps": len(records), "sim_duration_s": len(records) * DT, "termination_reason": termination, "completed_reference": len(records) == len(ref.q) - 1, "joint_peak_to_peak_rad": (q_array.max(axis=0) - q_array.min(axis=0)).tolist(), "joint_reference_peak_to_peak_rad": (ref.q.max(axis=0) - ref.q.min(axis=0)).tolist(), "ee_span_m": float(np.linalg.norm(ee.max(axis=0) - ee.min(axis=0))), "ee_path_length_m": float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()), "reference_ee_span_m": float(np.linalg.norm(ref_ee.max(axis=0) - ref_ee.min(axis=0))), "max_tracking_error_rad": float(max(r["tracking_error_rad"] for r in records)), "mean_tracking_error_rad": float(np.mean([r["tracking_error_rad"] for r in records])), "projection_fraction": float(np.mean([r["projected"] for r in records])), "wall_seconds": time.perf_counter() - start, "video_path": str(video_path) if video_path else None}
    return summary, records


def overlay(arr: np.ndarray, rows: list[dict], label: str, style: str) -> np.ndarray:
    pixels = np.asarray(arr).copy()
    # The integration MJCF intentionally has no presentation studio.  Apply a
    # documented post-render backdrop only where MuJoCo emitted near-black
    # clear color; robot geometry and all FK-derived overlays remain intact.
    background = pixels.max(axis=2) < 8
    if style == "light":
        yy = np.linspace(0, 1, pixels.shape[0])[:, None]
        studio = np.empty_like(pixels)
        studio[..., 0] = (231 - 18 * yy).astype(np.uint8)
        studio[..., 1] = (238 - 16 * yy).astype(np.uint8)
        studio[..., 2] = (242 - 10 * yy).astype(np.uint8)
        pixels[background] = studio[background]
    im = Image.fromarray(pixels).convert("RGB")
    if style == "dark":
        im = Image.fromarray((np.asarray(im, dtype=float) * np.array([0.62, 0.72, 0.84])).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA"); w, h = im.size
    d.rounded_rectangle((36, 34, 850, 205), radius=18, fill=(15, 28, 43, 220) if style == "dark" else (244, 248, 250, 225))
    fg = (236, 244, 250) if style == "dark" else (20, 38, 53)
    accent = (43, 190, 201)
    d.text((64, 57), "DEMO_ONLY  ·  EXPLORATORY VISUALIZATION", font=font(24, True), fill=(236, 173, 72))
    d.text((64, 95), label, font=font(31, True), fill=fg)
    last = rows[-1]
    d.text((64, 143), f"sim {last['sim_time_s']:5.2f} s   |   q-RMSE {last['tracking_error_rad']:.3f} rad   |   Safety QP {'PROJECTED' if last['projected'] else 'nominal'}", font=font(20), fill=accent)
    # A Cartesian XY inset is explicitly derived from FK for the reference and
    # actual state histories; it is not a decorative/synthetic trajectory.
    box = (w - 470, h - 355, w - 45, h - 45)
    d.rounded_rectangle(box, radius=18, fill=(12, 24, 38, 215), outline=(130, 170, 190, 130), width=2)
    d.text((box[0] + 22, box[1] + 17), "END-EFFECTOR XY · FK", font=font(19, True), fill=(240, 244, 247))
    actual = np.asarray([x["ee_actual_m"] for x in rows])[:, :2]
    ref = np.asarray([x["ee_reference_m"] for x in rows])[:, :2]
    pts = np.r_[actual, ref]; lo = pts.min(axis=0) - 0.02; hi = pts.max(axis=0) + 0.02; span = np.maximum(hi - lo, 0.05)
    def xy(a):
        x = box[0] + 25 + (a[:, 0] - lo[0]) / span[0] * (box[2] - box[0] - 50)
        y = box[3] - 28 - (a[:, 1] - lo[1]) / span[1] * (box[3] - box[1] - 78)
        return list(map(tuple, np.c_[x, y]))
    if len(ref) > 1: d.line(xy(ref), fill=(238, 178, 72, 255), width=4)
    if len(actual) > 1: d.line(xy(actual), fill=(52, 196, 210, 255), width=4)
    for name, color, xoff in (("REF", (238, 178, 72), 24), ("ACT", (52, 196, 210), 105)):
        d.ellipse((box[0]+xoff, box[3]-25, box[0]+xoff+12, box[3]-13), fill=color)
        d.text((box[0]+xoff+18, box[3]-31), name, font=font(15, True), fill=(235,240,242))
    return np.asarray(im)


def static_design_previews(frame: Path) -> None:
    im = Image.open(frame).convert("RGB")
    light = overlay(np.asarray(im), [{"sim_time_s": 0.0, "tracking_error_rad": 0.0, "projected": False, "ee_actual_m": [0, 0, 0], "ee_reference_m": [0, 0, 0]}], "Light Research Studio", "light")
    dark = overlay(np.asarray(im), [{"sim_time_s": 0.0, "tracking_error_rad": 0.0, "projected": False, "ee_actual_m": [0, 0, 0], "ee_reference_m": [0, 0, 0]}], "Dark Technical Studio", "dark")
    canvas = Image.new("RGB", (1920, 1080), (10, 17, 26)); canvas.paste(Image.fromarray(light).resize((960, 540)), (0, 0)); canvas.paste(Image.fromarray(dark).resize((960, 540)), (960, 0));
    d = ImageDraw.Draw(canvas)
    d.text((45, 600), "A  LIGHT RESEARCH STUDIO", font=font(31, True), fill=(227, 238, 246))
    d.text((1005, 600), "B  DARK TECHNICAL STUDIO", font=font(31, True), fill=(227, 238, 246))
    d.text((45, 654), "Preferred for GitHub: Panda silhouette and trace colors", font=font(20), fill=(180, 203, 216))
    d.text((45, 688), "remain legible on a white-page viewing context.", font=font(20), fill=(180, 203, 216))
    d.text((1005, 654), "Reserve for diagnostic clips: stronger contrast,", font=font(20), fill=(180, 203, 216))
    d.text((1005, 688), "but lower README-thumbnail clarity.", font=font(20), fill=(180, 203, 216))
    canvas.save(OUT / "light_dark_design_comparison_v2.png")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"append-only output directory exists: {OUT}")
    OUT.mkdir(parents=True)
    probe = make_env("pd_gravity_residual", Reference("probe", np.zeros((2, 7))))
    home = probe.home.copy(); model = probe.m; qadr = probe.s.qadr.copy(); qmin = probe.s.qmin.copy(); qmax = probe.s.qmax.copy(); probe.close()
    refs = [sweep(home), excitation(home), figure_eight(model, qadr, qmin, qmax, home)]
    feasibility = []
    for ref in refs:
        summary, rows = rollout("PD + analytical gravity · zero residual", "pd_gravity_residual", ref, None)
        feasibility.append(summary)
        (OUT / f"{ref.name}_pd_gravity_records.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Representative frozen-policy compatibility gate: all controller families,
    # seed 201, on the sweeping reference.  This is diagnostic only, not a
    # ranking and not a re-evaluation across 30 policies.
    mapping = {"pure_sac": ROOT / "policies/p4b/pure_sac/seed_201/best_model.zip", "pd_residual": ROOT / "policies/p4b/pd_residual/seed_201/best_model.zip", "mlp_residual": ROOT / "policies/p4b/mlp_residual/seed_201/best_model.zip", "gp_residual": ROOT / "policies/p4b/gp_residual/seed_201/best_model.zip", "rbf_residual": ROOT / "policies/p4b/rbf_residual/seed_201/best_model.zip", "pd_gravity_residual": ROOT / "policies/pd_gravity_posthoc/seed_201/best_model.zip"}
    compat = []
    for mode, path in mapping.items():
        policy = SAC.load(path, device="cpu")
        summary, _ = rollout(f"Frozen {mode} · seed 201", mode, refs[0], policy)
        summary["policy_path"] = str(path.relative_to(ROOT)); summary["policy_sha256"] = digest(path)
        compat.append(summary)
    # Render the longest safe controller/trajectory combination: the
    # zero-residual PD+Gravity baseline with 10 seconds of multi-joint
    # excitation.  This keeps the sample in the requested 10–15 s window.
    preview, rows = rollout("PD + analytical gravity · zero residual", "pd_gravity_residual", refs[1], None, render=True, style="light")
    (OUT / "preview_records.json").write_text(json.dumps(rows, indent=2) + "\n")
    first = imageio.get_reader(OUT / "v2a_motion_and_visual_style_preview.mp4").get_data(0)
    Image.fromarray(first).save(OUT / "v2a_preview_keyframe.png")
    # Short gif uses the first 4 seconds at 15 fps, preserving a documented
    # 1× simulation-to-video time mapping rather than time stretching.
    reader = imageio.get_reader(OUT / "v2a_motion_and_visual_style_preview.mp4")
    gif_frames = [Image.fromarray(x) for i, x in enumerate(reader) if i < 120 and i % 2 == 0]
    reader.close(); gif_frames[0].save(OUT / "v2a_preview.gif", save_all=True, append_images=gif_frames[1:], duration=67, loop=0, optimize=False)
    static_design_previews(OUT / "v2a_preview_keyframe.png")
    with (OUT / "feasibility.csv").open("w", newline="") as handle:
        fields = ["label", "mode", "controller_identity", "reference", "demo_classification", "steps", "sim_duration_s", "termination_reason", "completed_reference", "ee_span_m", "ee_path_length_m", "max_tracking_error_rad", "mean_tracking_error_rad", "projection_fraction", "wall_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{k: x[k] for k in fields} for x in feasibility + compat + [preview]])
    manifest = {"classification": "DEMO_ONLY / EXPLORATORY_VISUALIZATION", "formal_protocol_modified": False, "control_dt_s": DT, "physics_dt_s": 0.002, "video_fps": FPS, "video_time_scale": "1.0x: 10 seconds video equals 10 seconds simulated control time", "feasibility": feasibility, "representative_frozen_policy_compatibility": compat, "preview": preview, "files": {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

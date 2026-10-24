#!/usr/bin/env python3
"""V2-B1: offline, DEMO_ONLY studio rendering of a recorded V2-A episode.

This program never instantiates the control environment or advances physics.
It reads recorded q/qref states, performs FK in a separate visual-only model,
and renders visual-only floor/pedestal/lights plus 3-D trajectory overlays.
"""
from __future__ import annotations

import hashlib, json, math, shutil, subprocess, tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_propagation

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = BASE / "outputs_v2a_run04" / "A_multi_pose_sweeping_pd_gravity_records.json"
OUT = BASE / "outputs_v2b1" / "attempt_04"
FPS, DT, WIDTH, HEIGHT = 30, .004, 1920, 1080
START, END = 2.0, 6.0                 # selected from the completed 9.5-s episode
AMBER = np.array([0.96, 0.62, 0.13, 1.0], float)
CYAN = np.array([0.05, 0.78, 0.82, 1.0], float)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fnt(size: int, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + name, size)


def make_visual_xml(dst: Path, theme: str) -> None:
    """Copy source model then inject *non-contact* presentation-only geoms."""
    text = (ROOT / "assets/panda_direct_torque_v2.xml").read_text()
    text = text.replace('meshdir="mujoco_menagerie/franka_emika_panda/assets"',
                        'meshdir="../../../../assets/mujoco_menagerie/franka_emika_panda/assets"')
    if theme == "light":
        ambient, diffuse, spec = "0.55 0.55 0.55", "0.80 0.80 0.80", "0.20 0.20 0.20"
        floor, pedestal = "0.72 0.77 0.79 1", "0.18 0.24 0.28 1"
    else:
        ambient, diffuse, spec = "0.18 0.20 0.25", "0.65 0.68 0.78", "0.45 0.45 0.45"
        floor, pedestal = "0.035 0.055 0.075 1", "0.10 0.15 0.19 1"
    injected = f'''\n    <!-- V2-B1 VISUAL ONLY: contype/conaffinity zero; never used for physics. -->
    <geom name="v2b1_floor_visual" type="plane" size="3 3 .1" pos="0 0 -0.012" rgba="{floor}" contype="0" conaffinity="0"/>
    <geom name="v2b1_pedestal_visual" type="cylinder" size="0.255 0.035" pos="0 0 -0.035" rgba="{pedestal}" contype="0" conaffinity="0"/>
    <light name="v2b1_key_visual" pos="1.2 -1.1 2.2" dir="-1 1 -2" directional="true" ambient="{ambient}" diffuse="{diffuse}" specular="{spec}"/>
    <light name="v2b1_fill_visual" pos="-1.3 .6 1.4" dir="1 -.5 -1" directional="true" ambient=".08 .08 .08" diffuse=".32 .38 .45" specular=".05 .05 .05"/>
'''
    text = text.replace("  </worldbody>", injected + "  </worldbody>")
    dst.write_text(text)


def load_rows():
    rows = json.loads(SOURCE.read_text())
    return [r for r in rows if START - 1e-9 <= r["sim_time_s"] <= END + 1e-9]


def fk_series(model, rows):
    data = mujoco.MjData(model); qadr = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
    hand = model.body("hand").id
    actual, ref = [], []
    for row in rows:
        for q, dest in ((row["q"], actual), (row["qref"], ref)):
            data.qpos[qadr] = q; data.qvel[:] = 0; mujoco.mj_forward(model, data); dest.append(data.xpos[hand].copy())
    return qadr, np.asarray(actual), np.asarray(ref)


def add_capsule(scene, p0, p1, width, rgba):
    if scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.ones(3), np.zeros(3), np.eye(3).ravel(), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0), np.asarray(p1))
    scene.ngeom += 1


def add_sphere(scene, p, radius, rgba):
    if scene.ngeom >= scene.maxgeom: return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius]*3), np.asarray(p), np.eye(3).ravel(), rgba)
    scene.ngeom += 1


def camera(theme):
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Fixed oblique 45-degree composition, calibrated for the entire window.
    c.azimuth, c.elevation, c.distance = 132, -21, 1.72
    c.lookat = np.array([.07, .0, .48])
    return c


def draw_hud(frame, row, idx, actual, ref, theme):
    im = Image.fromarray(frame).convert("RGB"); d = ImageDraw.Draw(im, "RGBA"); w,h = im.size
    # Sidebar begins at 77.5% width: it never overlays main robot render.
    x0 = int(w*.775); bg = (12,25,36,238) if theme == "dark" else (241,246,247,244)
    fg = (239,246,250) if theme == "dark" else (20,39,51)
    muted = (158,184,198) if theme == "dark" else (76,104,117)
    d.rectangle((x0,0,w,h), fill=bg)
    d.rectangle((x0,0,x0+7,h), fill=(45,193,202,255))
    d.text((x0+34,54), "DEMO ONLY · SIMULATION", font=fnt(22,True), fill=(234,164,54))
    d.text((x0+34,104), "FRANKA PANDA", font=fnt(34,True), fill=fg)
    d.text((x0+34,151), "PD + Gravity", font=fnt(25,True), fill=fg)
    d.text((x0+34,186), "Zero SAC Residual", font=fnt(20), fill=muted)
    d.line((x0+34,234,w-34,234), fill=(*muted,100), width=2)
    d.text((x0+34,270), "RECORDED EPISODE", font=fnt(17,True), fill=muted)
    d.text((x0+34,301), f"t = {row['sim_time_s']:.2f} s", font=fnt(30,True), fill=fg)
    d.text((x0+34,367), "JOINT RMSE (rad)", font=fnt(17,True), fill=muted)
    d.text((x0+34,396), f"{row['tracking_error_rad']:.4f}", font=fnt(36,True), fill=(41,184,194))
    d.text((x0+34,474), "SAFETY QP", font=fnt(17,True), fill=muted)
    state = "NOMINAL · no projection" if not row['projected'] else "PROJECTED"
    d.text((x0+34,504), state, font=fnt(19,True), fill=(52,170,125) if not row['projected'] else (238,176,68))
    d.line((x0+34,562,w-34,562), fill=(*muted,100), width=2)
    d.text((x0+34,600), "3D END-EFFECTOR PATH", font=fnt(17,True), fill=muted)
    d.ellipse((x0+35,638,x0+51,654), fill=tuple((AMBER[:3]*255).astype(int)))
    d.text((x0+62,631), "Reference FK", font=fnt(18), fill=fg)
    d.ellipse((x0+35,677,x0+51,693), fill=tuple((CYAN[:3]*255).astype(int)))
    d.text((x0+62,670), "Actual FK", font=fnt(18), fill=fg)
    d.text((x0+34,h-72), "1× simulation time · visual replay", font=fnt(15), fill=muted)
    return np.asarray(im)


def studio_backdrop(frame, theme):
    """Replace only border-connected MuJoCo clear-color pixels, not robot black."""
    a=np.asarray(frame).copy(); candidate=a.max(axis=2)<10
    seed=np.zeros_like(candidate); seed[0,:]=candidate[0,:]; seed[-1,:]=candidate[-1,:]
    seed[:,0]|=candidate[:,0]; seed[:,-1]|=candidate[:,-1]
    mask=binary_propagation(seed,mask=candidate)
    h,w=mask.shape; yy=np.linspace(0,1,h)[:,None]
    if theme=='light': grad=np.dstack([226-20*yy,235-18*yy,239-12*yy]).astype(np.uint8)
    else: grad=np.dstack([14+7*yy,23+10*yy,32+15*yy]).astype(np.uint8)
    a[mask]=np.broadcast_to(grad,a.shape)[mask]
    return a


def render(theme, path, rows, model, qadr, actual, ref, tiny=False):
    H,W = (540,960) if tiny else (HEIGHT,WIDTH)
    model.vis.global_.offwidth=W; model.vis.global_.offheight=H
    renderer=mujoco.Renderer(model,height=H,width=W,max_geom=2000); data=mujoco.MjData(model); cam=camera(theme)
    frames=[]; indices=np.linspace(0,len(rows)-1,round((END-START)*FPS),dtype=int)
    for idx in indices:
        row=rows[idx]; data.qpos[qadr]=row['q']; data.qvel[:]=0; mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=cam)
        # Geometry below is renderer-only; it cannot influence contacts or state.
        # The record is 250 Hz but the visual curve is uniformly decimated to
        # at most 120 segments; this preserves the measured spatial path while
        # keeping a 1080p preview practical on the local GPU.
        trace_idx = np.unique(np.linspace(0, idx, min(121, idx + 1), dtype=int))
        for i0, i1 in zip(trace_idx[:-1], trace_idx[1:]):
            add_capsule(renderer.scene, ref[i0],ref[i1],.0042,AMBER)
            add_capsule(renderer.scene, actual[i0],actual[i1],.0054,CYAN)
        add_sphere(renderer.scene,ref[0],.012,AMBER); add_sphere(renderer.scene,actual[0],.010,CYAN)
        add_sphere(renderer.scene,ref[idx],.015,AMBER); add_sphere(renderer.scene,actual[idx],.013,CYAN)
        frame=studio_backdrop(renderer.render(),theme); frame=draw_hud(frame,row,idx,actual,ref,theme)
        if tiny: frames.append(frame)
        else: yield frame
    renderer.close()
    if tiny: return frames


def render_still(theme, row, idx, model, qadr, actual, ref):
    """A single independent Studio render used for Light/Dark comparison."""
    model.vis.global_.offwidth=WIDTH; model.vis.global_.offheight=HEIGHT
    renderer=mujoco.Renderer(model,height=HEIGHT,width=WIDTH,max_geom=2000); data=mujoco.MjData(model)
    data.qpos[qadr]=row['q']; data.qvel[:]=0; mujoco.mj_forward(model,data)
    renderer.update_scene(data,camera=camera(theme))
    trace_idx=np.unique(np.linspace(0,idx,min(121,idx+1),dtype=int))
    for i0,i1 in zip(trace_idx[:-1],trace_idx[1:]):
        add_capsule(renderer.scene,ref[i0],ref[i1],.0042,AMBER); add_capsule(renderer.scene,actual[i0],actual[i1],.0054,CYAN)
    add_sphere(renderer.scene,ref[0],.012,AMBER); add_sphere(renderer.scene,actual[0],.010,CYAN)
    add_sphere(renderer.scene,ref[idx],.015,AMBER); add_sphere(renderer.scene,actual[idx],.013,CYAN)
    frame=draw_hud(studio_backdrop(renderer.render(),theme),row,idx,actual,ref,theme); renderer.close()
    return frame


def contact_sheet(light, dark, out):
    canvas=Image.new('RGB',(1920,1080),(19,29,38))
    canvas.paste(Image.fromarray(light).resize((960,540)),(0,0)); canvas.paste(Image.fromarray(dark).resize((960,540)),(960,0))
    d=ImageDraw.Draw(canvas); d.text((44,600),'A  LIGHT RESEARCH STUDIO  ·  preferred',font=fnt(31,True),fill=(230,240,245))
    d.text((1004,600),'B  DARK TECHNICAL STUDIO',font=fnt(31,True),fill=(230,240,245))
    d.text((44,652),'Higher Panda, floor and amber/cyan trace clarity for white-page viewing.',font=fnt(20),fill=(176,201,213))
    d.text((1004,652),'Useful diagnostic alternative; reduced small-thumbnail legibility.',font=fnt(20),fill=(176,201,213))
    canvas.save(out)


def main():
    if OUT.exists(): raise SystemExit(f'append-only output exists: {OUT}')
    OUT.mkdir(parents=True); rows=load_rows()
    # Render models are generated locally and used only by this offline program.
    make_visual_xml(OUT/'studio_light_visual_only.xml','light'); make_visual_xml(OUT/'studio_dark_visual_only.xml','dark')
    light=mujoco.MjModel.from_xml_path(str(OUT/'studio_light_visual_only.xml')); dark=mujoco.MjModel.from_xml_path(str(OUT/'studio_dark_visual_only.xml'))
    qadr,actual,ref=fk_series(light,rows)
    # 1080p 1x continuous main preview; first frame is real recorded motion.
    video=OUT/'v2b1_studio_motion_preview.mp4'; writer=imageio.get_writer(video,fps=FPS,codec='libx264',quality=8,macro_block_size=1)
    for frame in render('light',video,rows,light,qadr,actual,ref): writer.append_data(frame)
    writer.close()
    # Keyframes are from independent Light and Dark scene renders at the window midpoint.
    midpoint=len(rows)//2
    lf=render_still('light',rows[midpoint],midpoint,light,qadr,actual,ref)
    df=render_still('dark',rows[midpoint],midpoint,dark,qadr,actual,ref)
    Image.fromarray(lf).resize((1920,1080)).save(OUT/'light_studio_keyframe.png'); Image.fromarray(df).resize((1920,1080)).save(OUT/'dark_studio_keyframe.png')
    contact_sheet(lf,df,OUT/'light_dark_studio_comparison.png')
    # Frame selection records retain source indices/timestamps and exact source hash.
    selected=[rows[int(i)] for i in np.linspace(0,len(rows)-1,round((END-START)*FPS),dtype=int)]
    (OUT/'trajectory_window_records.json').write_text(json.dumps(selected,indent=2)+'\n')
    np.savez_compressed(OUT/'trajectory_fk_world_m.npz', actual_m=actual,reference_m=ref,time_s=np.array([r['sim_time_s'] for r in rows]))
    command=['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,r_frame_rate,pix_fmt','-of','json',str(video)]
    ffprobe=shutil.which('ffprobe'); meta={'ffprobe_available':bool(ffprobe)}
    if ffprobe: meta['ffprobe']=json.loads(subprocess.check_output(command,text=True))
    manifest={'classification':'DEMO_ONLY / EXPLORATORY_VISUALIZATION','formal_physics_or_control_executed':False,'source_episode':'V2-A A_multi_pose_sweeping_pd_gravity_records.json','source_sha256':sha(SOURCE),'episode_controller':'PD + analytical gravity · zero SAC residual','episode_safe_completion':True,'window_sim_time_s':[START,END],'window_duration_s':END-START,'source_step_range':[rows[0]['step'],rows[-1]['step']],'control_dt_s':DT,'video_fps':FPS,'time_mapping':'1.0x; 4.0 s video equals 4.0 s recorded simulation time','visual_model_diff':'Only two non-contact geoms (floor/pedestal) and two lights were injected in copies. contype=conaffinity=0. No controller or physics data was loaded/stepped.','fk_end_effector_body':'hand','world_coordinate_system':'source MuJoCo world coordinates, metres','files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n'); (OUT/'media_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__': main()

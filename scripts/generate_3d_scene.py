"""
Generate an interactive 3D WebGL scene viewer from the MuJoCo simulation.
Uses brax.io.html to create a self-contained HTML with Three.js visualization.
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
import numpy as np
from mujoco import mjx
import jax.numpy as jnp
from brax.io import html as brax_html, json as brax_json, mjcf
from brax import base as brax_base

def generate_3d_scene(xml_path, out_path, Vx=10.0, R=0.3, Fz=4000, n_frames=200):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # === Load model for brax viewer (bypass collision validation) ===
    original_put_model = mjx.put_model
    def patched_put_model(m, device=None):
        orig_ct = m.geom_contype.copy()
        orig_ca = m.geom_conaffinity.copy()
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        try:
            return original_put_model(m, device)
        finally:
            m.geom_contype[:] = orig_ct
            m.geom_conaffinity[:] = orig_ca
    mjx.put_model = patched_put_model
    try:
        sys = mjcf.load_model(model)
    finally:
        mjx.put_model = original_put_model

    # === Patch brax's geom type names to support ellipsoid (render as Sphere) ===
    brax_json._GEOM_TYPE_NAMES[4] = 'Sphere'

    # === Get actuator IDs ===
    vel_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_x')
    spin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_spin')
    load_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'load_z')
    vel_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_y')
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'set_steer')
    camber_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'set_camber')

    kappas = np.linspace(-0.3, 0.3, n_frames)

    # Warmup
    for _ in range(1000):
        data.ctrl[load_z_id] = -Fz
        data.ctrl[vel_x_id] = Vx
        data.ctrl[vel_y_id] = 0
        data.ctrl[steer_id] = 0
        data.ctrl[camber_id] = 0
        data.ctrl[spin_id] = Vx * (1 + kappas[0]) / R
        mujoco.mj_step(model, data)

    # Record frames
    states = []
    for kappa in kappas:
        for _ in range(20):
            data.ctrl[load_z_id] = -Fz
            data.ctrl[vel_x_id] = Vx
            data.ctrl[vel_y_id] = 0
            data.ctrl[steer_id] = 0
            data.ctrl[camber_id] = 0
            data.ctrl[spin_id] = Vx * (1 + kappa) / R
            mujoco.mj_step(model, data)

        # Record body transforms (skip worldbody at index 0)
        pos = jnp.array(data.xpos[1:])
        rot = jnp.array(data.xquat[1:])
        x = brax_base.Transform(pos=pos, rot=rot)
        states.append(brax_base.State(x=x, xd=None, q=None, qd=None, contact=None))

    # Render
    html_str = brax_html.render(sys, states, height=600, colab=False)
    with open(out_path, 'w') as f:
        f.write(html_str)
    print(f"3D scene saved to {out_path} ({len(states)} frames)")


if __name__ == '__main__':
    xml_path = os.path.abspath('/home/bairg/MjTire/assets/virtual_test_rig.xml')
    out_path = '/home/bairg/MjTire/results/scene_3d.html'
    generate_3d_scene(xml_path, out_path)

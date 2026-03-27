import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
import numpy as np
import plotly.graph_objects as go
import imageio
from datetime import datetime

def main():
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/virtual_test_rig.xml'))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    vel_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_x')
    spin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_spin')
    load_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'load_z')
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'set_steer')
    vel_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_y')

    spin_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin')
    trans_x_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'translate_x')

    Vx = 10.0 # m/s
    R = 0.3   # m
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    def step_and_render(step_count, record_video):
        mujoco.mj_step(model, data)
        # Render at 30 FPS
        if record_video and step_count % 33 == 0:
            renderer.update_scene(data, camera="chase_cam")
            frames.append(renderer.render())

    timesteps = int(4.0 / model.opt.timestep)
    kappas = np.linspace(-0.3, 0.3, timesteps)
    
    fz_loads = [2000, 4000, 6000]
    fx_results = {}
    actual_kappas_all = {}

    for Fz in fz_loads:
        print(f"Sweeping for Fz = {Fz} N")
        record_video = (Fz == 4000)
        mujoco.mj_resetData(model, data)
        step_counter = 0

        # Long Sweep Warmup
        for _ in range(int(1.0 / model.opt.timestep)):
            data.ctrl[load_z_id] = -Fz
            data.ctrl[vel_x_id] = Vx
            data.ctrl[vel_y_id] = 0
            data.ctrl[steer_id] = 0
            data.ctrl[spin_id] = Vx * (1 + kappas[0]) / R 
            step_and_render(step_counter, record_video)
            step_counter += 1

        # Long sweep
        fx_list = []
        act_kappas = []
        for kappa in kappas:
            omega = Vx * (1 + kappa) / R
            data.ctrl[load_z_id] = -Fz
            data.ctrl[spin_id] = omega
            step_and_render(step_counter, record_video)
            step_counter += 1
            fx_list.append(-data.actuator_force[vel_x_id])
            act_kappas.append(kappa)
            
        fx_results[Fz] = fx_list
        actual_kappas_all[Fz] = act_kappas

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(__file__), f'../results/phase1/{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Plot Fx
    fig = go.Figure()
    for Fz in fz_loads:
        fig.add_trace(go.Scatter(x=actual_kappas_all[Fz], y=fx_results[Fz], mode='lines', line=dict(width=3), name=f'Fz = {Fz}N'))
    
    fig.update_layout(title="MuJoCo Default Contact: Longitudinal Force vs Slip Ratio",
                      xaxis_title="Longitudinal Slip Ratio (\u03BA)",
                      yaxis_title="Longitudinal Force Fx (N)",
                      legend_title_text="Vertical Load",
                      height=600, width=800,
                      template="plotly_white")
    
    out_path_fx_html = os.path.join(output_dir, 'baseline_fx.html')
    fig.write_html(out_path_fx_html)
    print("Saved HTML plot to", out_path_fx_html)

    # Save Video
    out_video_path = os.path.join(output_dir, 'baseline_sweep.gif')
    print("Saving video to", out_video_path)
    imageio.mimsave(out_video_path, frames, fps=30)
    print("Done!")

if __name__ == '__main__':
    main()

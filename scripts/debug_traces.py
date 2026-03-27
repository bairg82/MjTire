import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    xml_path = os.path.abspath('assets/virtual_test_rig.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    vel_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_x')
    spin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_spin')
    load_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'load_z')
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'set_steer')
    vel_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_y')
    camber_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'set_camber')

    spin_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'spin')
    steer_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'steer')
    camber_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'camber')
    steer_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'steer_link')

    Vx = 10.0  # m/s
    R = 0.3    # m
    n_kappa_points = 200
    dwell_steps = 20  # steps to settle at each kappa before recording
    kappas = np.linspace(-0.3, 0.3, n_kappa_points)
    Fz = 4000

    t_list = []
    vx_list = []
    vy_list = []
    omega_list = []
    kappa_list = []
    alpha_list = []  # lateral slip angle
    gamma_list = []  # camber angle

    mujoco.mj_resetData(model, data)

    def record():
        t_list.append(data.time)
        res = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, steer_body, res, 1)
        local_vx = res[3]
        local_vy = res[4]
        vx_list.append(local_vx)
        vy_list.append(local_vy)

        omega = data.qvel[model.jnt_dofadr[spin_jnt]]
        omega_list.append(omega)

        # Longitudinal slip ratio: kappa = (omega*R - Vx) / |Vx|
        if abs(local_vx) > 0.1:
            k = (omega * R - local_vx) / abs(local_vx)
        else:
            k = 0.0
        kappa_list.append(k)

        # Lateral slip angle: alpha = -atan2(Vy, |Vx|)
        if abs(local_vx) > 0.1:
            alpha = -np.arctan2(local_vy, abs(local_vx))
        else:
            alpha = 0.0
        alpha_list.append(np.degrees(alpha))

        # Camber angle from joint position
        gamma = data.qpos[model.jnt_qposadr[camber_jnt]]
        gamma_list.append(np.degrees(gamma))

    def step_with_ctrl(kappa_cmd):
        data.ctrl[load_z_id] = -Fz
        data.ctrl[vel_x_id] = Vx
        data.ctrl[vel_y_id] = 0
        data.ctrl[steer_id] = 0.0
        data.ctrl[camber_id] = 0.0
        data.ctrl[spin_id] = Vx * (1 + kappa_cmd) / R
        mujoco.mj_step(model, data)

    # Warmup — settle at first kappa
    for _ in range(int(1.0 / model.opt.timestep)):
        step_with_ctrl(kappas[0])

    # Sweep with dwell: settle at each kappa, then record the settled state
    for kappa in kappas:
        for _ in range(dwell_steps):
            step_with_ctrl(kappa)
        record()

    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=('Local Longitudinal Speed (Wheel Frame)',
                                        'Local Lateral Speed (Wheel Frame)',
                                        'Angular Velocity',
                                        'Longitudinal Slip Ratio (\u03BA)',
                                        'Lateral Slip Angle (\u03B1)',
                                        'Camber Angle (\u03B3)'))

    fig.add_trace(go.Scatter(x=t_list, y=vx_list, mode='lines', name='Vx'), row=1, col=1)
    fig.update_yaxes(title_text='Vx (m/s)', row=1, col=1)

    fig.add_trace(go.Scatter(x=t_list, y=vy_list, mode='lines', name='Vy'), row=2, col=1)
    fig.update_yaxes(title_text='Vy (m/s)', row=2, col=1)

    fig.add_trace(go.Scatter(x=t_list, y=omega_list, mode='lines', name='Omega'), row=3, col=1)
    fig.update_yaxes(title_text='Omega (rad/s)', row=3, col=1)

    fig.add_trace(go.Scatter(x=t_list, y=kappa_list, mode='lines', name='Slip Ratio', line=dict(color='orange', width=2)), row=4, col=1)
    fig.update_yaxes(title_text='\u03BA', row=4, col=1)

    fig.add_trace(go.Scatter(x=t_list, y=alpha_list, mode='lines', name='Slip Angle', line=dict(color='green', width=2)), row=5, col=1)
    fig.update_yaxes(title_text='\u03B1 (deg)', row=5, col=1)

    fig.add_trace(go.Scatter(x=t_list, y=gamma_list, mode='lines', name='Camber', line=dict(color='red', width=2)), row=6, col=1)
    fig.update_yaxes(title_text='\u03B3 (deg)', row=6, col=1)
    fig.update_xaxes(title_text='Time (s)', row=6, col=1)

    fig.update_layout(height=1400, width=1000, showlegend=False,
                      title_text='Pure Kinematics Test (Zero Friction)')

    out_path = '/home/bairg/MjTire/results/time_traces.html'
    fig.write_html(out_path)

    print("Done")

if __name__ == '__main__':
    main()

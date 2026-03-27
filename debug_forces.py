import mujoco
import numpy as np
model = mujoco.MjModel.from_xml_path('assets/virtual_test_rig.xml')
data = mujoco.MjData(model)
load_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'load_z')

print("Fz_ctrl | Actual Fn | load_z_force | Carriage_Z")
for Fz in [2000, 4000, 6000]:
    mujoco.mj_resetData(model, data)
    for _ in range(500):
        data.ctrl[load_z_id] = -Fz
        mujoco.mj_step(model, data)
        
    Fn = 0
    for i in range(data.ncon):
        res = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, res)
        Fn += res[0]
        
    carriage_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'carriage')
    carriage_z = data.xpos[carriage_id][2]
    print(f"{Fz} | {Fn:.2f} | {data.actuator_force[load_z_id]:.2f} | {carriage_z:.4f}")

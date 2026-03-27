import unittest
import mujoco
import numpy as np
import os

class TestVirtualRigKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/virtual_test_rig.xml'))
        
    def setUp(self):
        # Fresh model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

    def test_01_compilation(self):
        """Ensure the XML compiles without structural or semantic errors."""
        self.assertIsNotNone(self.model, "Model failed to load/compile.")
        self.assertGreater(self.model.nbody, 0, "Model has no bodies.")

    def test_02_carriage_translation(self):
        """Test driving the carriage purely in X axis without pitch/yaw/roll."""
        vel_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_x')
        self.data.ctrl[vel_x_id] = 5.0
        
        for _ in range(int(1.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            
        carriage_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'carriage')
        # cvel represents [angular(3), linear(3)] spatial velocity
        cvel = self.data.cvel[carriage_id] 
        
        # The carriage should not be pitching or yawing while purely translating
        np.testing.assert_allclose(cvel[:3], [0, 0, 0], atol=1e-3, err_msg="Carriage has unexpected angular velocity (wobble/pitching) during pure X translation.")
        # It shouldn't drift sideways
        self.assertAlmostEqual(cvel[4], 0.0, delta=1e-3, msg="Carriage drifting laterally (Y) when driven purely in X.")
        
    def test_03_wheel_spin_kinematics(self):
        """Apply spin torque to the wheel in mid-air and check for translational or off-axis rotational wobble."""
        # Lift the rig so we test pure mid-air spin without ground contact interference
        load_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'load_z')
        self.data.ctrl[load_z_id] = 5000  # Pull up
        
        # Spin wheel
        spin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'drive_spin')
        self.data.ctrl[spin_id] = 10.0 # 10 rad/s
        
        for _ in range(int(1.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            
        wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'wheel')
        cvel = self.data.cvel[wheel_id]
        omega_x, omega_y, omega_z = cvel[:3]
        
        # The wheel should be spinning firmly around its defined Y-axis
        self.assertGreater(abs(omega_y), 5.0, "Wheel is not spinning sufficiently around Y-axis.")
        
        # But it should absolutely not wobble around X or Z
        self.assertAlmostEqual(omega_x, 0.0, delta=1e-3, msg="FAIL: Wheel angular wobble around X-axis (Pitch).")
        self.assertAlmostEqual(omega_z, 0.0, delta=1e-3, msg="FAIL: Wheel angular wobble around Z-axis (Yaw).")

    def test_04_numerical_stability(self):
        """Run the simulation under gravity only and check for exploding velocities (NaNs)."""
        for _ in range(int(2.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.assertFalse(np.isnan(self.data.qvel).any(), "NaN found in qvel, indicating model explosion (singular inertia matrix or excessive forces).")
            
        # Check collision depths
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            self.assertGreater(contact.dist, -0.05, f"Excessive continuous penetration depth {contact.dist} detected between geoms.")

    def test_06_contact_patch_kinematics(self):
        """The contact patch site must NOT rotate with the wheel. It should stay at the bottom."""
        # Topologically, the site must be attached to the steer_link, not the wheel body.
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'contact_patch')
        self.assertNotEqual(site_id, -1, "contact_patch site not found!")
        
        body_id = self.model.site_bodyid[site_id]
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        self.assertEqual(body_name, 'steer_link', f"FAIL: contact_patch is attached to '{body_name}'. It MUST be on 'steer_link' to avoid spinning.")

    def test_07_geom_axis_alignment(self):
        """A cylinder geom's symmetry axis MUST align with the spin joint axis, otherwise it tumbles!"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'tire')
        # By default, a MuJoCo cylinder's symmetry axis is its local Z-axis (0, 0, 1)
        # We need to find the geom's orientation relative to the body
        geom_quat = self.model.geom_quat[geom_id]
        
        # Calculate local Z vector of the geom
        # Standard rotation of (0,0,1) by a quaternion [w, x, y, z] is [2(xz+wy), 2(yz-wx), 1-2(x^2+y^2)]
        w, x, y, z = geom_quat
        local_z_of_geom = np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x**2 + y**2)])
        
        # Check against spin axis
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'spin')
        spin_axis = self.model.jnt_axis[joint_id]
        
        # They must be parallel (dot product magnitude should be 1)
        dot_prod = abs(np.dot(local_z_of_geom, spin_axis))
        self.assertAlmostEqual(dot_prod, 1.0, delta=1e-3, msg=f"FAIL: Tumbling geometry! Cylinder symmetry axis {local_z_of_geom} is NOT aligned with spin axis {spin_axis}.")

if __name__ == '__main__':
    unittest.main()

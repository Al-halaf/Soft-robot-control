import numpy as np
import mujoco
import mujoco.viewer
import time

# Load the MuJoCo model and data
model = mujoco.MjModel.from_xml_path("experiment.xml")
data = mujoco.MjData(model)

# Get the ID of the "sBase" site and "tennis_ball" body
site_name = "sBase"
body_name = "tennis_ball"
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

def move_arm_to_position(target_position, viewer, tolerance=1e-1, max_steps=500, learning_rate=0.1):
    """
    Move the robot arm to the target position using numerical inverse kinematics.
    """
    for step in range(max_steps):
        # Compute the current end-effector position
        mujoco.mj_forward(model, data)
        end_effector_pos = data.site(site_id).xpos

        # Calculate the error between target and current position
        error = target_position - end_effector_pos
        error_norm = np.linalg.norm(error)

        print(f"Step {step}: End effector position = {end_effector_pos}, Error = {error_norm}")

        if error_norm < tolerance:
            print("Target reached!")
            break

        # Compute the Jacobian matrix
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # Update joint angles using gradient descent
        delta_q = learning_rate * jacp.T @ error
        data.ctrl[:6] += delta_q[:6]

        # Step the simulation and add a small delay
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # Slow down the movement

def adjust_wrist_orientation(viewer, max_steps=5, learning_rate=0.02):
    """
    Automatically adjust the wrist orientation for a better grip.
    """
    for step in range(max_steps):
        mujoco.mj_forward(model, data)
        end_effector_quat = data.xquat[site_id]
        desired_quat = np.array([1, 1, 0, 0])  # Example: Adjust as needed
        error = desired_quat - end_effector_quat

        if np.linalg.norm(error) < 1e-1:
            print("Wrist aligned!")
            break

        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, None, jacr, site_id)
        delta_q = learning_rate * jacr.T @ error[:3]
        data.ctrl[:6] += delta_q[:6]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


def apply_grip_force(viewer, max_force=-5.0, max_steps=1000, tolerance=1e-2):
    """
    Apply force-based control to the gripper for adaptive gripping.
    """
    for step in range(max_steps):
        mujoco.mj_forward(model, data)
        contact_forces = data.sensordata[:4]  # Assuming there are 4 finger force sensors

        for i in range(4):
            finger_pos = data.qpos[6 + i]  # Get current finger position
            target_direction = np.sign(data.body(body_id).xpos - data.site(site_id).xpos)[0]  # Ensure scalar value
            movement_direction = np.sign(data.qvel[6 + i])  # Current movement direction

            if contact_forces[i] > max_force:
                print(f"Finger {i}: Excessive force detected, reducing grip pressure.")
                data.ctrl[6 + i] -= 0.02  # Loosen grip slightly
            elif contact_forces[i] < tolerance:
                if movement_direction != target_direction:
                    print(f"Finger {i}: Moving away from ball, reversing direction.")
                    data.ctrl[6 + i] -= 0.01 * target_direction  # Reverse movement
                else:
                    print(f"Finger {i}: No contact detected, increasing grip pressure.")
                    data.ctrl[6 + i] += 0.01  # Tighten grip slightly
            else:
                print(f"Finger {i}: Grip is stable.")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


def open_gripper(viewer):
    for _ in range(100):
        data.ctrl[6:10] = 0.2  # Open gripper
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # Slow down the gripper opening
    data.ctrl[6:10] = 0
    viewer.sync()


def lift_arm(viewer, lift_height=0.5):
    target_pos = data.site(site_id).xpos + np.array([0, 0, lift_height])
    move_arm_to_position(target_pos, viewer)

def pick_up_tennis_ball(viewer):
    # Step 1: Open Gripper
    print("Opening gripper...")
    open_gripper(viewer)

    # Step 2: Move directly above the Tennis Ball
    tennis_ball_pos = data.body(body_id).xpos
    target_pos = np.array(
        [tennis_ball_pos[0], tennis_ball_pos[1], tennis_ball_pos[2] + 0.5])  # Move directly above the ball
    print(f"Moving to position above the ball: {target_pos}")
    move_arm_to_position(target_pos, viewer)
    adjust_wrist_orientation(viewer)

    for _ in range(100):
        continue

    # Step 3: Lower the arm onto the Tennis Ball
    target_pos = tennis_ball_pos + np.array([0, 0, 0.15])  # Slightly above the ball
    print(f"Lowering to position: {target_pos}")
    move_arm_to_position(target_pos, viewer)

    # Step 4: Close Gripper
    print("Closing gripper...")
    apply_grip_force(viewer, max_force=0.5)

    # Step 5: Lift the ball
    print("Lifting arm...")
    lift_arm(viewer)

# Run the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Initialize
    for _ in range(100):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Execute pick-up
    pick_up_tennis_ball(viewer)

    # Keep viewer open indefinitely until manually closed
    print("Simulation complete. Close the viewer window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # Prevent high CPU usage

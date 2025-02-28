import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

class SoftGripperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode="human"):
        super().__init__()

        # Load MuJoCo model
        self.model_path = "experiment.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)  # Mujoco data object
        self.max_steps = 10000
        self.current_step = 0
        self.prev_distance = 0

        # Define action space (6-DOF arm + gripper control)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Define observation space (joint positions + object position)
        num_joints = self.model.nq
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_joints * 2 + 5,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None

    def step(self, action):
        """Apply action, advance simulation, return obs, reward, done, info"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action  # Apply action
        mujoco.mj_step(self.model, self.data)

        # Increment step counter
        self.current_step += 1

        # Get ball and gripper positions
        ball_id = self.model.body("tennis_ball").id
        gripper_id = self.model.body("base").id
        ball_pos = self.data.xpos[ball_id]
        gripper_pos = self.data.xpos[gripper_id]

        # Compute distance
        distance = np.linalg.norm(gripper_pos - ball_pos)

        # Compute reward
        reward = -distance  # Encourage moving closer to the ball

        # Reward for reducing the distance to the ball
        if distance < self.prev_distance:
            reward += 5 # Encourage getting closer
        else:
            reward -= 5  # Penalize moving away from the ball

        # Bonus reward if the gripper is very close
        if distance < 0.1:
            reward += 2

        if distance < 0.05:
            reward += 6

        if distance < 0.02:
            reward += 10

            # Extra reward if the gripper moves above the ball (to encourage grasping)
        if gripper_pos[2] > ball_pos[2] + 0.02:
            reward += 25

            # Store the current distance for the next step
        self.prev_distance = distance

        # Done conditions
        terminated = distance < 0.01  # Success if gripper is very close
        truncated = self.current_step >= self.max_steps  # Stop after max steps

        # Observation
        obs = np.concatenate([self.data.qpos, self.data.qvel, ball_pos, gripper_pos])

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset simulation and return initial observation"""
        self.current_step = 0  # Reset step counter
        self.data.qpos[:] = 0  # Reset all joint positions

        # Set ball's initial position
        ball_joint_id = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")]
        self.data.qpos[ball_joint_id: ball_joint_id + 3] = [0.5, 0, 0.05]

        mujoco.mj_forward(self.model, self.data)  # Ensure new position is applied

        # Compute the observation
        obs = np.concatenate([
            self.data.qpos, self.data.qvel,
            self.data.xpos[self.model.body("tennis_ball").id],
            self.data.xpos[self.model.body("base").id]
        ])
        return obs, {}

    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)  # Correct MuJoCo viewer
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            return self.data.qpos.copy()  # You may need a proper image capture function

    def close(self):
        """Cleanup"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Soft-robot-control
This project focuses on training a soft robotic gripper to pick up a tennis ball using reinforcement learning (RL) and MuJoCo. By leveraging Stable-Baselines3, this environment allows an RL agent to learn grasping behaviors through trial and error.

🚀 Features
✅ MuJoCo-based Robotic Simulation – A custom-built MuJoCo environment for training a robotic arm with a soft gripper.
✅ Reinforcement Learning with PPO – Uses Proximal Policy Optimization (PPO) from Stable-Baselines3 to train the agent.
✅ Reward Optimization – Encourages reaching, grasping, and lifting the ball while penalizing unnecessary movement.
✅ Training Checkpointing – Allows saving and continuing training from previous sessions.
✅ Interactive Rendering – Visualizes the agent’s performance in real time.

📦 Requirements
To run the project, install the following dependencies:

🛠 Install Required Libraries
bash
Copy
Edit
pip install stable-baselines3 gymnasium[mujoco] mujoco numpy
📋 Required Software
Python (≥3.8 recommended)
MuJoCo (DeepMind's version)
🔄 Workflow
1️⃣ Environment Setup – Loads the MuJoCo model and initializes the RL environment.
2️⃣ Training the RL Agent – Uses PPO to train the soft gripper to grasp the ball.
3️⃣ Reward Shaping

✅ Positive rewards for moving toward the ball.
✅ Bonus for grasping and lifting the ball.
❌ Penalty for moving away from the ball.
4️⃣ Training Continuation – Allows loading saved models to continue training.
5️⃣ Testing – Runs a trained policy in the simulation to evaluate performance.
🎯 Applications
This project can be applied in:

🤖 Robotic grasping research for industrial and household robots.
🏗 Reinforcement learning experiments in continuous control tasks.
🧩 Soft robotics simulations for real-world deployment.
Reinforcement learning experiments in continuous control tasks.
Soft robotics simulations for real-world deployment.

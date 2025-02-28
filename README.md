# 🤖 Soft Gripper RL Simulation (Work in Progress)

This project focuses on **training a soft robotic gripper** to pick up a **tennis ball** using **reinforcement learning (RL)** and **MuJoCo**. By leveraging **Stable-Baselines3**, this environment allows an RL agent to learn grasping behaviors through trial and error.

---

## 🚀 Features

- **MuJoCo-based Robotic Simulation** – A custom-built MuJoCo environment for training a robotic arm with a soft gripper.  
- **Reinforcement Learning with PPO** – Uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3 to train the agent.  
- **Reward Optimization** – Encourages reaching, grasping, and lifting the ball while penalizing unnecessary movement.  
- **Training Checkpointing** – Allows saving and continuing training from previous sessions.  
- **Interactive Rendering** – Visualizes the agent’s performance in real time.  

---

## 📦 Requirements

To run the project, install the following dependencies:

### Install Required Libraries

```bash
pip install stable-baselines3 gymnasium[mujoco] mujoco numpy
```

### Required Software

- **Python (>=3.8 recommended)**  
- **MuJoCo (DeepMind's version)**  

---

## 🔄 Workflow

1. **Environment Setup** – Loads the MuJoCo model and initializes the RL environment.  
2. **Training the RL Agent** – Uses **PPO** to train the soft gripper to grasp the ball.  
3. **Reward Shaping**  
   - ✅ Positive rewards for moving toward the ball.  
   - ✅ Bonus for grasping and lifting the ball.  
   - ❌ Penalty for moving away from the ball.  
4. **Training Continuation** – Allows loading saved models to continue training.  
5. **Testing** – Runs a trained policy in the simulation to evaluate performance.  

---

## 🎯 Applications

This project can be applied in:

- 🤖 **Robotic grasping research** for industrial and household robots.  
- 🏗 **Reinforcement learning experiments** in continuous control tasks.  
- 🧩 **Soft robotics simulations** for real-world deployment.  

---

## ⚡ Example Usage

### **Train the Model**

```python
from stable_baselines3 import PPO
env = SoftGripperEnv(render_mode=None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save("soft_gripper_policy")
```

### **Continue Training**

```python
model = PPO.load("soft_gripper_policy", env=env)
model.learn(total_timesteps=2_000_000)
model.save("soft_gripper_policy_v2")
```

### **Test the Trained Model**

```python
env = SoftGripperEnv(render_mode="human")
model = PPO.load("soft_gripper_policy_v2")
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
env.close()
```

---

## 📢 Work in Progress

This project is **actively being developed**, with improvements including:

- 🔄 **Fine-tuning the reward function** for better grasping behavior.  
- 🔬 **Exploring alternative RL algorithms (SAC, TD3) for better control.**  
- 📊 **Adding visualization tools to track training performance.**  

---

## 🤝 Contributions & Feedback

If you have ideas or improvements, feel free to **open an issue** or **submit a pull request**! 🚀


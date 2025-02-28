from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym_env

# Create the environment
env = DummyVecEnv([lambda: gym_env.SoftGripperEnv(render_mode=None)])  # Wrap in vectorized environment


# Load the trained model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, ent_coef=0.01)

# Train the model
TIMESTEPS = 20_000_000   # Adjust based on your needs
model.learn(total_timesteps=TIMESTEPS)

# Save the trained model
model.save("soft_gripper_policy")

print("Training complete!")

# Create environmentx
env = gym_env.SoftGripperEnv(render_mode="human")

model = PPO.load("soft_gripper_policy")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)  # Get action from trained policy
    obs, reward, done, truncated, _ = env.step(action)
    env.render()

env.close()
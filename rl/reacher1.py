import gymnasium as gym
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def make_env():
    env = gym.make("Reacher-v5", render_mode="human")
    env = Monitor(env, log_dir)
    return env

env = DummyVecEnv([make_env])

# PPO parameters
ppo_params = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64, # bigger batch size for faster training
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95, # Generalized Advantage Estimation
    "ent_coef": 0.0,
    "clip_range": 0.2,
    "verbose": 1
}

# neural network architecture for policy and value function
# pi = policy (actor), vf = value function (critic)
# so, for each of the two networks, we have 2 hidden layers with 256 neurons each
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

# Multi Layer Perceptron Policy with custom architecture
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    **ppo_params)

# Create evaluation callback, saves the best model
eval_callback = EvalCallback(
    env,
    best_model_save_path=f"{model_dir}/best_model",
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False
)

total_timesteps = 100000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True)

# Save the model
model.save(f"{model_dir}/ppo_reacher")


obs = env.reset()
episodes = 10
total_rewards = []
for ep in range(episodes):
    done = False
    episode_reward = 0
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render("human")
        
        time.sleep(0.01)
    total_rewards.append(episode_reward)
    print(f"Episode {ep + 1}: Total Reward: {episode_reward}")
    obs - env.reset()
print(f"Average Reward over {episodes} episodes: {sum(total_rewards) / episodes}")
env.close()

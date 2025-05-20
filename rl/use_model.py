import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time

def make_env():
    return gym.make("Reacher-v5", render_mode="human")

# Create and wrap the environment
env = DummyVecEnv([make_env])

# Load the trained model
model_path = "models/best_model/best_model"  # adjust path if needed
model = PPO.load(model_path)

# Test the model
episodes = 5
total_rewards = []

for episode in range(episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Get model's prediction
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute action in environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]
        env.render("human")
        
        # Small delay to better visualize the movement
        time.sleep(0.01)
    
    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

print(f"\nAverage Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
env.close()
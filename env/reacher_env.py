import gymnasium as gym
import matplotlib.pyplot as plt
import time

# SIMPLE TEST
env = gym.make("Reacher-v5", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Replace with your controller's action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
    
    time.sleep(0.01) 

env.close()

import gymnasium as gym
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO

# SIMPLE TEST
# adding render_mode="human" means that the environment will be rendered in a window
# but will take longer to run
# rgb array is faster and can be used to save frames
# ansi is a text-based rendering mode

env = gym.make("Reacher-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)

# Multi Layer Perceptron Policy
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


vec_env = model.get_env()
obs = vec_env.reset()
episodes = 10
for ep in range(episodes):
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        
        time.sleep(0.01) 

env.close()

import gymnasium as gym
import matplotlib.pyplot as plt
import time
from stable_baselines3 import A2C

# SIMPLE TEST
# adding render_mode="human" means that the environment will be rendered in a window
# but will take longer to run
# rgb array is faster and can be used to save frames
# ansi is a text-based rendering mode

env = gym.make("Reacher-v5", render_mode="rgb_array")
obs, info = env.reset(seed=42)

# Multi Layer Perceptron Policy
model = A2C("MlpPolicy", env, verbose=1)
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


'''
REACHER-V5 info
sample action shape: (2,)
sample observation space: (10,)
Continuous action space with 2 dimensions

rl training output info
rollout/
ep_len_mean - average length of episodes over the recent window (default: 100)
ep_rew_mean - average reward per episode over the recent window (default: 100) if rising, there is progress
time/
iterations - number of training iterations completed
time_elapsed - total time elapsed since the start of training [s]
total_timesteps - total number of timesteps collected since the start of training
train/
entropy_loss - entropy loss, higher values means more randomness and exploration. 
as training progresses, this value should decrease
n_updates - number of training updates performed
policy_loss - policy gradient loss


'''
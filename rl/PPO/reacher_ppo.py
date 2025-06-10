import gymnasium as gym
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ALGO_DIR, "models")
LOG_DIR = os.path.join(ALGO_DIR, "logs")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)


def make_env():
    env = gym.make("Reacher-v5")
    env = Monitor(env, LOG_DIR)
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
    "ent_coef": 0.01,
    "clip_range": 0.2, # how much the policy can change
    "verbose": 1
}

# neural network architecture for policy and value function
# pi = policy (actor), vf = value function (critic)
# so, for each of the two networks, we have 2 hidden layers with 256 neurons each
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

# Multi Layer Perceptron Policy with custom architecture
# Using CPU for PPO is faster than GPU
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=TENSORBOARD_DIR,
    **ppo_params, device="cpu")

eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join(ALGO_DIR, "best_model"),    
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=MODEL_DIR,
    name_prefix="ppo_reacher_checkpoint"
)

total_timesteps = 600000
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True)

# Save the model
model.save(os.path.join(MODEL_DIR, "ppo_reacher_final"))



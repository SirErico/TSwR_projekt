import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import os
from typing import Callable

# Set up directories
ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ALGO_DIR, "models")
LOG_DIR = os.path.join(ALGO_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env() -> Callable:
    """Create a Reacher environment with monitoring."""
    env = gym.make("Reacher-v5")
    env = Monitor(env, LOG_DIR)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

sac_params = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",  # Automatic entropy tuning
    "verbose": 1
}

# Neural network architecture for policy and value function
policy_kwargs = dict(
    net_arch=dict(
        pi=[400, 300],  # Actor network
        qf=[400, 300]   # Critic network
    )
)

# Initialize SAC model
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    **sac_params
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Train the model
total_timesteps = 50000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
    progress_bar=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "sac_reacher_final"))
env.close()

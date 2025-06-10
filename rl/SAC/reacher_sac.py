import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from typing import Callable

# Set up directories
ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ALGO_DIR, "models")
LOG_DIR = os.path.join(ALGO_DIR, "logs")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

def make_env() -> Callable:
    """Create a Reacher environment with monitoring."""
    env = gym.make("Reacher-v5")
    env = Monitor(env, LOG_DIR)
    return env

def linear_schedule(initial_value: float, final_value: float):
    """Linear learning rate schedule."""
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

# Create vectorized environment
env = DummyVecEnv([make_env])

sac_params = {
    "learning_rate": linear_schedule(3e-4, 1e-4),
    "buffer_size": 1_000_000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005, # target smoothing coef
    "gamma": 0.99, # discount factor
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
    tensorboard_log=TENSORBOARD_DIR,
    device="cuda",
    **sac_params
)

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
    name_prefix="sac_reacher_checkpoint"
)

# Train the model
total_timesteps = 600000
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "sac_reacher_final"))
env.close()

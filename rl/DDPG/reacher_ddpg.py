import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
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

# Create vectorized environment
env = DummyVecEnv([make_env])
n_actions = env.action_space.shape[0]

# DDPG parameters
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

ddpg_params = {
    "learning_rate": 1e-3,
    "buffer_size": 1000000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
    "action_noise": action_noise,
    "verbose": 1
}

# Neural network architecture for policy and value function
policy_kwargs = dict(
    net_arch=dict(
        pi=[400, 300],  # Actor network
        qf=[400, 300]   # Critic network
    )
)

# Initialize DDPG model
model = DDPG(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=TENSORBOARD_DIR,
    **ddpg_params
)

# Create evaluation callback
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

# Train the model
total_timesteps = 600000
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "ddpg_reacher_final"))
env.close()

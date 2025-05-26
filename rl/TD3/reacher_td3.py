import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
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

# Action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)
)

# TD3 parameters
td3_params = {
    "learning_rate": 1e-3,
    "buffer_size": 1_000_000,
    "learning_starts": 10000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_delay": 2,  # TD3-specific: delayed policy updates
    "target_policy_noise": 0.2,  # TD3-specific: target policy smoothing
    "target_noise_clip": 0.5,  # TD3-specific: noise clipping
    "train_freq": (1, "episode"),
    "action_noise": action_noise,
    "verbose": 1
}

# Neural network architecture
policy_kwargs = dict(
    net_arch=dict(
        pi=[400, 300],  # Actor network
        qf=[400, 300]   # Critic networks (TD3 uses twin critics)
    )
)

# Initialize TD3 model
model = TD3(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=TENSORBOARD_DIR,
    device="cuda",
    **td3_params
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=10000,
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=MODEL_DIR,
    name_prefix="td3_reacher_checkpoint"
)
# Train the model
total_timesteps = 700000  
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)

# Save final model
model.save(os.path.join(MODEL_DIR, "td3_reacher_final"))
env.close()
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import os

# Set up directories
ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ALGO_DIR, "models/best_model/best_model")

def evaluate_model(model: SAC, env: gym.Env, episodes: int = 10) -> None:
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained SAC model
        env: Gymnasium environment
        episodes: Number of episodes to evaluate
    """
    total_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            time.sleep(0.01)  #
            
        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")
    env.close()
    print(f"\nAverage Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")

def main():
    # Create and wrap the environment
    env = gym.make("Reacher-v5", render_mode="human")
    
    # Load the trained model
    try:
        model = SAC.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    # Evaluate the model
    evaluate_model(model, env)


if __name__ == "__main__":
    main()
    
# -5.32
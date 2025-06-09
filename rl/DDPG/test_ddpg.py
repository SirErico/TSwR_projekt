import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import os
from typing import Callable
import matplotlib.pyplot as plt

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ALGO_DIR, "models/best_model/best_model_DDPG_2")

def get_end_effector_pos(env):
    """Get end-effector position from environment."""
    l1, l2 = 0.1, 0.1  # Link lengths from MuJoCo model
    q1, q2 = env.unwrapped.data.qpos[:2]
    
    # Forward kinematics
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return np.array([x, y])

def evaluate_model(model: DDPG, env: gym.Env, episodes: int = 10) -> None:
    total_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Initialize tracking lists
        ee_positions = []
        target_positions = []
        distances = []
        torques = []
        torques_sqr = []
        times = []
        
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            torques.append(action)
            torques_sqr.append(action ** 2)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Track data
            ee_pos = get_end_effector_pos(env)
            target_pos = obs[4:6]  # Target position from observation
            
            ee_positions.append(ee_pos)
            target_positions.append(target_pos)
            distances.append(np.linalg.norm(ee_pos - target_pos))
            times.append(steps)
            steps += 1
            
            if abs(obs[8]) < 0.01 and abs(obs[9]) < 0.01:
                print("num of steps: ", steps)
                done = True
            episode_reward += reward
            time.sleep(0.01)  #
            
        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")

        # Convert to numpy arrays for plotting
        ee_positions = np.array(ee_positions)
        target_positions = np.array(target_positions)
        distances = np.array(distances)
        torques = np.array(torques)
        torques_sqr = np.array(torques_sqr)
        times = np.array(times)
        
        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # (0, 0): End-effector trajectory
        axs[0, 0].plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', label='End-effector')
        axs[0, 0].plot(target_positions[:, 0], target_positions[:, 1], 'r*', label='Target')
        axs[0, 0].set_title('End-effector Trajectory')
        axs[0, 0].set_xlabel('X Position')
        axs[0, 0].set_ylabel('Y Position')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # (0, 1): Distance to target over time
        axs[0, 1].plot(times, distances, 'g-')
        axs[0, 1].set_title('Distance to Target')
        axs[0, 1].set_xlabel('Time Step')
        axs[0, 1].set_ylabel('Distance')
        axs[0, 1].grid(True)

        # (1, 0): Torques applied over time
        axs[1, 0].plot(times, torques[:, 0], label='Joint 1')
        axs[1, 0].plot(times, torques[:, 1], label='Joint 2')
        axs[1, 0].set_title('Torques Applied')
        axs[1, 0].set_xlabel('Time Step')
        axs[1, 0].set_ylabel('Torque')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # (1, 1): Cumulative torque usage
        cumulative_torque = np.cumsum(np.abs(torques), axis=0)
        axs[1, 1].plot(times, cumulative_torque[:, 0], label='Cumulative Joint 1')
        axs[1, 1].plot(times, cumulative_torque[:, 1], label='Cumulative Joint 2')
        axs[1, 1].set_title('Cumulative Torque Over Time')
        axs[1, 1].set_xlabel('Time Step')
        axs[1, 1].set_ylabel('Cumulative |Torque|')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(ALGO_DIR, f'episode_{ep + 1}_analysis.png'))

    env.close()
    print(f"\nAverage Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")

def main():
    # Added seeding
    SEED = 42
    env = gym.make("Reacher-v5", render_mode="human") # max_episode_steps
    env.reset(seed=SEED)
    try:
        model = DDPG.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    evaluate_model(model, env)

if __name__ == "__main__":
    main()
    
# -4.92

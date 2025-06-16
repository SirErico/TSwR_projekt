import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ALGO_DIR, "best_model_PPO_2")
ALGORITHM = PPO
ALGO_NAME = ALGORITHM.__name__

# Configuration
SEED = 42
MAX_STEPS = 150
LINK_LENGTH = 0.1 # link lengths (0.1) from mujoco docs 

def get_end_effector_pos(env, l1=LINK_LENGTH, l2=LINK_LENGTH):
    """Get end-effector position from environment."""
    q1, q2 = env.unwrapped.data.qpos[:2]
    
    # Forward kinematics
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return np.array([x, y])

def evaluate_model(model: ALGORITHM, env: gym.Env, episodes: int = 10) -> None:
    episode_data = []
    total_rewards = []
    dt = env.unwrapped.model.opt.timestep
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        steps = 0
        
        # Initialize tracking lists
        episode_distances = []
        episode_torques = []
        episode_torques_sqr = [] # is it necessary?
        episode_powers = []
        
        while not done:
            q_dot = env.unwrapped.data.qvel[:2].copy()
            action, _state = model.predict(obs, deterministic=True)
            
            # Append data
            episode_torques.append(action)
            episode_torques_sqr.append(action ** 2)
            power = np.abs(np.dot(action, q_dot))
            episode_powers.append(power)
            
            # Calculate energy cost
            step_cost = power * dt
            episode_cost += step_cost
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Track data
            ee_pos = get_end_effector_pos(env)
            target_pos = obs[4:6]
            dist = np.linalg.norm(ee_pos - target_pos)
            episode_distances.append(dist)
            # print("DISTANCE = ", dist)
            # done = terminated or truncated

            
            steps += 1
            episode_reward += reward
            
            # Update done flag based on conditions
            done = (steps >= MAX_STEPS) or (dist < 0.01 and np.all(np.abs(q_dot) < 0.1))
            if dist < 0.01 and np.all(np.abs(q_dot) < 0.1):
                print(f"Episode {ep+1}: Reached the goal in {steps} steps.")
            elif steps >= MAX_STEPS:
                print(f"Episode {ep+1}: Max steps reached without convergence.")
            time.sleep(0.01)  #
        episode_data.append({
            'episode': ep + 1,
            'steps': steps,
            'reward': episode_reward,
            'energy_cost': episode_cost,
        })
        total_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")
        
        # Convert to numpy arrays for plotting
        time_steps = np.arange(len(episode_torques))
        episode_torques = np.array(episode_torques)
        episode_powers = np.array(episode_powers)
        
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Plot torques
        axes[0].plot(time_steps, episode_torques[:, 0], 'b-', label='Joint 1')
        axes[0].plot(time_steps, episode_torques[:, 1], 'r-', label='Joint 2')
        axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        axes[0].axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_title(f'Episode {ep+1}: Applied Torques')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Torque (N⋅m)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot distances
        axes[1].plot(time_steps, episode_distances, 'g-')
        axes[1].axhline(y=0.01, color='k', linestyle='--', alpha=0.3)
        axes[1].set_title(f'Episode {ep+1}: Distance to Target')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Distance (m)')
        axes[1].grid(True)
        
        # Plot power
        axes[2].plot(time_steps, episode_powers, 'm-')
        axes[2].set_title(f'Episode {ep+1}: Instantaneous Power')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Power (W)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ALGO_DIR, f'{ALGO_NAME}_analysis_episode_{ep + 1}.png'))
        plt.close()   
    env.close()  
      
    # Create summary plots
    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(ALGO_DIR, f'{ALGO_NAME}_results.csv'), index=False)
    
    # Energy cost bar chart
    plt.figure(figsize=(12, 6))
    episodes_range = np.arange(1, episodes + 1)
    energy_costs = [data['energy_cost'] for data in episode_data]

    plt.bar(episodes_range, energy_costs, color='skyblue', edgecolor='navy')
    plt.title(f"{ALGO_NAME} Energy Cost per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Energy Cost (J)")

    for i, cost in enumerate(energy_costs):
        plt.text(i + 1, cost, f'{cost:.2f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ALGO_DIR, f'{ALGO_NAME}_energy_cost_barchart.png'))
    plt.close()
    
    # Print statistics
    print("\nSummary Statistics:")
    print(f"Average Energy Cost: {np.mean(energy_costs):.2f}")
    print(f"Energy Cost Std: {np.std(energy_costs):.2f}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Reward Std: {np.std(total_rewards):.2f}")

def main():
    print(f"Using algorithm: {ALGO_NAME}")  
    env = gym.make("Reacher-v5", render_mode="human") # max_episode_steps
    env.reset(seed=SEED)
    try:
        model = ALGORITHM.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    evaluate_model(model, env)

if __name__ == "__main__":
    main()


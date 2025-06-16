import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
SEED = 42
MAX_STEPS = 50
LINK_LENGTH = 0.1 # link lengths (0.1) from mujoco docs 
GLOBAL_T = [0.25]
KP = np.diag([30, 30])
KD = np.diag([10, 10])
ALPHA, BETA, GAMMA = 0.00263468, 0.00009852, 0.0013667

def M_hat(q, alpha, beta, gamma):
    q2 = q[1]
    m11 = alpha + 2*beta*np.cos(q2)
    m12 = gamma + beta*np.cos(q2)
    m21 = gamma + beta*np.cos(q2)
    m22 = gamma      
    return np.array([[m11, m12], [m21, m22]])

def C_hat(q, q_dot, alpha, beta, gamma):
    q2 = q[1]
    q1_dot = q_dot[0]
    q2_dot = q_dot[1]
    c11 = -beta*np.sin(q2)*q2_dot
    c12 = -beta*np.sin(q2)*(q1_dot+q2_dot)
    c21 = beta*np.sin(q2)*q1_dot
    c22 = 0
    return np.array([[c11, c12], [c21, c22]])

def inverse_kinematics(xy_target, l1=LINK_LENGTH, l2=LINK_LENGTH):
    x, y = xy_target
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if np.abs(D) > 1.0:
        return None  # unreachable
    q2 = np.arccos(D)
    k2 = l2 * np.sin(q2)
    k1 = l1 + l2 * np.cos(q2)
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return np.array([q1, q2])

def get_end_effector_pos(q, l1= LINK_LENGTH, l2=LINK_LENGTH):
    """Calculate end-effector position using forward kinematics."""  # Link lengths
    q1, q2 = q
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return np.array([x, y])

def generate_trajectory(q_start, q_goal, T, dt):
    steps = int(T / dt)
    time_array = np.linspace(0, T, steps)
    q_r = np.zeros((steps, 2))
    q_r_dot = np.zeros((steps, 2))
    q_r_ddot = np.zeros((steps, 2))
    
    for i, t in enumerate(time_array):
        s = t / T
        s2 = s ** 2
        s3 = s ** 3

        # Cubic polynomial interpolation (q_dot(0) = q_dot(T) = 0)
        # no acceleration
        h = q_goal - q_start
        q_r[i] = q_start + h * (3 * s2 - 2 * s3)
        q_r_dot[i] = h * (6 * s - 6 * s2) / T
        q_r_ddot[i] = h * (6 - 12 * s) / (T**2)
    return q_r, q_r_dot, q_r_ddot


def feedback_linearization_control():
    np.set_printoptions(legacy='1.25') # Print bez  float64
    # ESTIMATED VALUES
    # alpha, beta, gamma = 0.00263468, 0.00009852, 0.0013667 # 0.00263468, 0.00009852, 0.00013667 / 0.00254362, 0.00001777, 0.00029786
    SEED = 1

    env = gym.make("Reacher-v5", max_episode_steps=150, render_mode="human")
    env.reset(seed=SEED)
    # could get the target from the wrapper
    # goal_pos = env.unwrapped.get_body_com("target")

    dt = env.unwrapped.model.opt.timestep
    print("DT",dt) # 0.01

    Kp = KP
    Kd = KD
    episodes = 10
    
    # TESTING
    T_values = GLOBAL_T # Different time durations for trajectory generation
    
    # Initialize tracking lists
    episode_data = []
    all_torques = []
    all_torques_sqr = []
    all_powers = []
    total_rewards = []

    for T in T_values:
        for ep in range(episodes):
            episode_distances = []
            episode_torques_noclip = []
            episode_torques = []
            episode_torques_sqr = []
            episode_powers = []
            episode_reward = 0
            episode_cost = 0.0
            steps = 0
            torque_violations = 0
            obs, _ = env.reset()
            
            done = False
            goal = (obs[4], obs[5])
            ik_result = inverse_kinematics(goal)
            if ik_result is None:
                print(f"T={T}, Episode {ep+1}: Goal unavailable!")
                continue

            
            q_start = env.unwrapped.data.qpos[:2].copy()
            q_r_all, q_r_dot_all, q_r_ddot_all = generate_trajectory(q_start, ik_result, T, dt=dt)
            """
            0.1 definitely too fast. tau gets too big
            around 0.25?
            """
            while not done and steps < MAX_STEPS:
                q = env.unwrapped.data.qpos[:2].copy()
                q_dot = env.unwrapped.data.qvel[:2].copy()
                
                if steps < len(q_r_all):
                    q_r = q_r_all[steps]
                    q_r_dot = q_r_dot_all[steps]
                    q_r_ddot = q_r_ddot_all[steps]
                else:
                    q_r = ik_result
                    q_r_dot = np.zeros(2)
                    q_r_ddot = np.zeros(2)

                v = q_r_ddot + Kd @ (q_r_dot - q_dot) + Kp @ (q_r - q)  # control
                # v = q_r_dot
                # Compute dynamics
                M = M_hat(q, ALPHA, BETA, GAMMA)
                C = C_hat(q, q_dot, ALPHA, BETA, GAMMA)

                # Final torque
                tau_noclip = M @ v + C @ q_dot
                if np.any(np.abs(tau_noclip) > 1.0):
                    torque_violations += 1
                    print(f"Episode {ep + 1}, Step {steps}: Torque violation: {tau_noclip}")
                
                # Liczenie kosztu energetycznego (energia =  |tau ⋅ q_dot| * dt)
                power = np.abs(np.dot(tau_noclip, q_dot))
                step_cost = power * dt
                episode_cost += step_cost
                
                # Clip to action space limits
                tau = np.clip(tau_noclip, env.action_space.low, env.action_space.high)

                    
                episode_torques_noclip.append(tau_noclip)  # Store unclipped torque
                episode_torques.append(abs(tau))
                episode_torques_sqr.append(tau ** 2)
                episode_powers.append(power)

                obs, reward, terminated, truncated, info = env.step(tau)
                ee_pos = get_end_effector_pos(q)
                target_pos = np.array([obs[4], obs[5]])  # Correct target position from observation
                dist = np.linalg.norm(ee_pos - target_pos)
                episode_distances.append(dist)
                
                steps += 1
                episode_reward += reward
  
                #done = terminated or truncated
                if dist < 0.01 and np.all(np.abs(q_dot) < 0.1):
                    print(f"T={T}, Episode {ep+1}: Reached the goal in {steps} steps.")
                    done = True
                elif steps >= MAX_STEPS:
                    print(f"T={T}, Episode {ep+1}: Max steps reached without convergence.")
                    done = True

                time.sleep(0.01)  #

            episode_data.append({
                'T': T,
                'episode': ep + 1,
                'steps': steps,
                'reward': episode_reward,
                'energy_cost': episode_cost,
                'torque_violations': torque_violations
            })
            total_rewards.append(episode_reward)  # Add this line to track rewards
        
            print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")
            # Here plots for each episode :)
            
            time_steps = np.arange(len(episode_torques_noclip))
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            
            # Plot unclipped torques
            axes[0].plot(time_steps, np.array(episode_torques_noclip)[:, 0], 'b-', label='Joint 1')
            axes[0].plot(time_steps, np.array(episode_torques_noclip)[:, 1], 'r-', label='Joint 2')
            axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
            axes[0].axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
            axes[0].set_title(f'T={T}, Episode {ep+1}: Unclipped Torques')
            axes[0].set_xlabel('Time Steps')
            axes[0].set_ylabel('Torque (N⋅m)')
            axes[0].legend()
            axes[0].grid(True)
            
            # Distance
            axes[1].plot(time_steps, episode_distances, 'g-')
            axes[1].axhline(y=0.01, color='k', linestyle='--', alpha=0.3)
            axes[1].set_title(f'T={T}, Episode {ep+1}: Distance')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Distance (m)')
            axes[1].grid(True)
            
            # Power
            axes[2].plot(time_steps, episode_powers, 'm-')
            axes[2].set_title(f'T={T}, Episode {ep+1}: Instantaneous Power')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Power (W)')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(ALGO_DIR, f'analysis_{T}_torques_episode_{ep + 1}.png'))
            plt.close()

            # for i, j in episode_torques_noclip:
            #     if abs(i) >= 1.0 or abs(j) >= 1.0:
            #         print(f"Episode {ep + 1} has torques exceeding limits: {i}, {j}")

        env.close()

    df = pd.DataFrame(episode_data)
    df.to_csv(os.path.join(ALGO_DIR, 'feedback_linearization_results.csv'), index=False)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='T', y='energy_cost', data=df)
    plt.title('Energy Cost vs. Trajectory Duration')
    plt.xlabel('T (s)')
    plt.ylabel('Energy Cost')
    plt.grid(True)
    plt.savefig(os.path.join(ALGO_DIR, 'energy_boxplot.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='T', y='torque_violations', data=df)
    plt.title('Torque Violations vs. Trajectory Duration')
    plt.xlabel('T (s)')
    plt.ylabel('Number of Violations')
    plt.grid(True)
    plt.savefig(os.path.join(ALGO_DIR, 'torque_violations_boxplot.png'))
    plt.close()
    
    # Create bar chart for episode costs
    plt.figure(figsize=(12, 6))
    episodes_range = np.arange(1, episodes + 1)

    # Group energy costs by T value
    for i, T in enumerate(T_values):
        T_costs = [data['energy_cost'] for data in episode_data if data['T'] == T]
        offset = i * (1.0/len(T_values))  # Calculate offset for grouped bars
        x_positions = episodes_range + offset
        
        plt.bar(x_positions, 
                T_costs, 
                width=1.0/len(T_values), 
                label=f'T={T}s',
                alpha=0.8)

        plt.title("Energy Cost per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Energy Cost (J)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ALGO_DIR, f'energy_cost_barchart_{T}.png'))
        plt.close()
    
    print("\nSummary Statistics:")
    print(df.groupby('T')[['reward', 'energy_cost', 'torque_violations']].mean())
    if total_rewards:
        print(f"\nAverage Reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}")
        print(f"Standard Deviation: {np.std(total_rewards):.2f}")
    else:
        print("\nNo valid rewards collected")  
        
if __name__ == "__main__":
    feedback_linearization_control()
    
"""
# check different T and the smallest taus/q_ddot?
# tau nie wychodzi poza 1 (10 testow)

zużycie energii całkowite
całkowanie mocy
boxploty (tau, tau**, moc, czas)
pobrać csv

"""
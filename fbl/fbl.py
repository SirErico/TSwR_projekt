import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

ALGO_DIR = os.path.dirname(os.path.abspath(__file__))

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

# link lengths (0.1) from mujoco docs 
def inverse_kinematics(xy_target, l1=0.1, l2=0.1):
    x, y = xy_target
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if np.abs(D) > 1.0:
        return None  # unreachable
    q2 = np.arccos(D)
    k2 = l2 * np.sin(q2)
    k1 = l1 + l2 * np.cos(q2)
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return np.array([q1, q2])

def get_end_effector_pos(q):
    """Calculate end-effector position using forward kinematics."""
    l1, l2 = 0.1, 0.1  # Link lengths
    q1, q2 = q
    x = l1*np.cos(q1) + l2*np.cos(q1 + q2)
    y = l1*np.sin(q1) + l2*np.sin(q1 + q2)
    return np.array([x, y])

def ret_y(y, speed1, speed2 = -50000.0):
    if speed2 == -50000.0:
        speed2 = speed1
    if y < 0:
        return np.array([-speed1, -speed2])
    else:
        return np.array([speed1, speed2])

def q_r_ddot_simple_traj(dist, y, q_r_dot, q_r, breaking, speed, steps, first_step, max_step):
    # Breaking
    """
    if 0.02 < dist < breaking:
        current_step = steps - first_step
        current_speed = speed - ((current_step / max_step) * speed)
        out_speed = (current_speed - speed) / current_step
        return (ret_y(y, out_speed))
    """
    # Near target
    if dist < 0.021:
        return np.array([0,0])
    else:
        sp1 = 0.0
        sp2 = 0.0
        # Acceleration
        if abs(q_r_dot[0] - q_r[0]) < 0.1:
            sp1 = speed
        # Enough speed
        if abs(q_r_dot[1] - q_r[1]) < 0.1:
            sp2 = speed
        return ret_y(y, sp1, sp2)

def q_r_dot_simple_traj(dist, y, steps, first_step, max_step, breaking, speed):
    """
    if 0.02 < dist < breaking:
        current_step = steps - first_step
        current_speed = speed - ((current_step / max_step) * speed)
        return (ret_y(y, current_speed))
    """
    if dist < 0.021:
        return np.array([0,0])
    else:
        return ret_y(y, speed)

# check different T and the smallest taus/q_ddot?
# tau nie wychodzi poza 1 (10 testow)
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
    alpha, beta, gamma = 0.00263468, 0.00009852, 0.0013667 # 0.00263468, 0.00009852, 0.00013667 / 0.00254362, 0.00001777, 0.00029786
    SEED = 1

    env = gym.make("Reacher-v5", max_episode_steps=150, render_mode="human")
    env.reset(seed=SEED)
    # could get the target from the wrapper
    # goal_pos = env.unwrapped.get_body_com("target")

    dt = env.unwrapped.model.opt.timestep

    Kp = np.diag([30, 30])
    Kd = np.diag([10, 10])
    episodes = 10
    
    # Initialize tracking lists
    torques = []
    torques_sqr = []
    total_rewards = []
    episode_costs = []

    for ep in range(episodes):
        distances = []
        episode_torques_noclip = []
        episode_reward = 0
        episode_cost = 0.0
        steps = 0
        obs, _ = env.reset()
        
        done = False
        goal = (obs[4], obs[5])
        ik_result = inverse_kinematics(goal)
        if ik_result is None:
            print("Goal unavailable!")
            continue
        print("\nTarget position: ")
        print(obs[8], obs[9])
        
        q_start = env.unwrapped.data.qpos[:2].copy()
        q_r_all, q_r_dot_all, q_r_ddot_all = generate_trajectory(q_start, ik_result, T=0.25, dt=dt)
        """
        0.1 definitely too fast. tau gets too big
        around 0.25?
        """
        
        while not done:
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
            M = M_hat(q, alpha, beta, gamma)
            C = C_hat(q, q_dot, alpha, beta, gamma)

            # Final torque
            tau_noclip = M @ v + C @ q_dot
            episode_torques_noclip.append(tau_noclip)  # Store unclipped torque
            
            # Liczenie kosztu energetycznego (energia = ∑ |tau ⋅ q_dot| * dt)
            step_cost = np.abs(np.dot(tau_noclip, q_dot)) * dt
            episode_cost += step_cost

            # Clip to action space limits
            tau = np.clip(tau_noclip, env.action_space.low, env.action_space.high)
            torques.append(abs(tau))
            torques_sqr.append(tau ** 2)

            obs, reward, terminated, truncated, info = env.step(tau)
            ee_pos = get_end_effector_pos(q)
            target_pos = np.array([obs[4], obs[5]])  # Correct target position from observation
            dist = np.linalg.norm(ee_pos - target_pos)
            distances.append(dist)
            """
            if terminated or truncated:
                print("How close to the target: x, y")
                print(obs[8], obs[9])
            """
            #done = terminated or truncated
            if np.linalg.norm([obs[8], obs[9]]) < 0.01 and abs(q_dot[0]) < 0.1 and abs(q_dot[1]) < 0.1:
                print(f"Reached the goal in {steps} steps.")
                done = True
            steps += 1
            time.sleep(0.01)  #
            episode_reward += reward

            # if terminated or truncated:
            #     break
        total_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")
        # Here plots for each episode :)
        
        # After episode ends, create plots
        episode_torques_noclip = np.array(episode_torques_noclip)
        time_steps = np.arange(len(episode_torques_noclip))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot unclipped torques
        ax1.plot(time_steps, episode_torques_noclip[:, 0], 'b-', label='Joint 1')
        ax1.plot(time_steps, episode_torques_noclip[:, 1], 'r-', label='Joint 2')
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=-1.0, color='k', linestyle='--', alpha=0.3)
        ax1.set_title('Unclipped Torques Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Torque (N⋅m)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot DISTANCE
        ax2.plot(time_steps, distances, 'g-')
        ax2.set_title('Distance to Target')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ALGO_DIR, f'torques_episode_{ep + 1}.png'))
        plt.close()

        for i, j in episode_torques_noclip:
            if abs(i) >= 1.0 or abs(j) >= 1.0:
                print(f"Episode {ep + 1} has torques exceeding limits: {i}, {j}")

    env.close()

    # Create bar chart for episode costs
    plt.figure(figsize=(12, 6))
    episodes_range = np.arange(1, episodes + 1)
    plt.bar(episodes_range, episode_costs, color='skyblue', edgecolor='navy')
    plt.title("Koszt energetyczny (τ·ω) dla każdego epizodu")
    plt.xlabel("Numer epizodu")
    plt.ylabel("Całkowity koszt energetyczny")
    
    # Add value labels on top of each bar
    for i, cost in enumerate(episode_costs):
        plt.text(i + 1, cost, f'{cost:.2f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ALGO_DIR, 'energy_cost_barchart.png'))
    plt.close()

    print(f"\nAverage Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")
    
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
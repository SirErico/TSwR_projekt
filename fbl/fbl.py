import gymnasium as gym
import numpy as np
import time

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


def feedback_linearization_control():
    np.set_printoptions(legacy='1.25') # Print bez  float64
    # ESTIMATED VALUES
    alpha, beta, gamma = 0.00263468, 0.00009852, 0.0013667 # 0.00263468, 0.00009852, 0.00013667 / 0.00254362, 0.00001777, 0.00029786

    env = gym.make("Reacher-v5", max_episode_steps=150, render_mode="human")
    # could get the target from the wrapper
    # goal_pos = env.unwrapped.get_body_com("target")

    dt = env.unwrapped.model.opt.timestep

    q_r = np.array([0.0, 0.0]) 
    q_r_dot = np.zeros(2)         
    q_r_ddot = np.zeros(2) 

    Kp = np.diag([10, 10])
    Kd = np.diag([5, 5])
    episodes = 10
    for ep in range(episodes):
        episode_reward = 0
        steps = 0
        obs, _ = env.reset()
        done = False
        goal = (obs[4], obs[5])
        ik_result = inverse_kinematics(goal)
        if ik_result is None:
            tau = np.zeros(2)
        else:
            q_r = ik_result
        print("\nTarget position: ")
        print(goal)
        
        while not done:
            steps += 1
            q = env.unwrapped.data.qpos[:2].copy()
            q_dot = env.unwrapped.data.qvel[:2].copy()

            v = q_r_ddot + Kd @ (q_r_dot - q_dot) + Kp @ (q_r - q)  # control

            # Compute dynamics
            M = M_hat(q, alpha, beta, gamma)
            C = C_hat(q, q_dot, alpha, beta, gamma)

            # Final torque
            tau = M @ v + C @ q_dot

            # Clip to action space limits
            tau = np.clip(tau, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(tau)
            """
            if terminated or truncated:
                print("How close to the target: x, y")
                print(obs[8], obs[9])
            """
            # done = terminated or truncated
            # if close to traget or low velocity, then done
            if abs(obs[8]) < 0.01 and abs(obs[9]) < 0.01 and abs(q_dot[0]) < 0.1 and abs(q_dot[1]) < 0.1:
                print("num of steps: ", steps)
                done = True
            time.sleep(0.01)  #
            episode_reward += reward

            # if terminated or truncated:
            #     break
        print(f"Episode {ep + 1}: Total Reward: {episode_reward:.2f}")

    env.close()
    
if __name__ == "__main__":
    feedback_linearization_control()
    
"""
Dodać trajektorie
jak szybko znalezione 
jaki był koszt

"""
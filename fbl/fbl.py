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

def feedback_linearization_control():
    # ESTIMATED VALUES
    alpha, beta, gamma = 1.0, 0.5, 0.5

    env = gym.make("Reacher-v5", render_mode="human")
    # could get the target from the wrapper
    # goal_pos = env.unwrapped.get_body_com("target")

    dt = env.unwrapped.model.opt.timestep

    q_r = np.array([0.0, 0.0]) 
    q_r_dot = np.zeros(2)          
    q_r_ddot = np.zeros(2)    

    Kp = np.diag([30, 30])
    Kd = np.diag([20, 20])
    episodes = 10
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
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
            done = terminated or truncated
            time.sleep(0.01)  #
            # if terminated or truncated:
            #     break

    env.close()
    
if __name__ == "__main__":
    feedback_linearization_control()
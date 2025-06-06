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

    env = gym.make("Reacher-v5", max_episode_steps=150, render_mode="human")
    # could get the target from the wrapper
    # goal_pos = env.unwrapped.get_body_com("target")

    dt = env.unwrapped.model.opt.timestep

    Kp = np.diag([30, 30])
    Kd = np.diag([10, 10])
    episodes = 10
    for ep in range(episodes):
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
        q_r_all, q_r_dot_all, q_r_ddot_all = generate_trajectory(q_start, ik_result, T=1.5, dt=dt)
        
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
            # print("\nTarget position: ")
            # print(goal)
            # print("V: ")
            # print(v)
            # Compute dynamics
            M = M_hat(q, alpha, beta, gamma)
            C = C_hat(q, q_dot, alpha, beta, gamma)

            # Final torque
            tau = M @ v + C @ q_dot
            # print("Tau: ")
            # print(tau)

            # Clip to action space limits
            tau = np.clip(tau, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(tau)
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
            # if terminated or truncated:
            #     break

    env.close()
    
if __name__ == "__main__":
    feedback_linearization_control()
import gymnasium as gym
import numpy as np
import pandas as pd
import math
import csv

def get_samples():
    env = gym.make("Reacher-v5", render_mode="none")
    obs, _ = env.reset(seed=0)

    data = []

    data_dict = {
        'q1': [], 'q2': [], 
        'q1_dot': [], 'q2_dot': [], 
        'q1_ddot': [], 'q2_ddot': [], 
        'tau1': [], 'tau2': []
    }
    
    prev_qvel = None
    dt = env.unwrapped.model.opt.timestep

    for _ in range(5000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt        
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()

    for t in range(10000):
        torque1 = math.sin(t * 0.05)
        torque2 = math.cos(t * 0.05)
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()

    for t in range(10000):
        torque1 = math.cos(t * 0.05)
        torque2 = math.sin(t * 0.05)
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()
    
    for t in range(5000):
        torque1 = 1.0
        torque2 = 1.0
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()
    
    for t in range(5000):
        torque1 = -1.0
        torque2 = -1.0
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            qacc = (qvel - prev_qvel) / dt
            
            data_dict['q1'].append(qpos[0])
            data_dict['q2'].append(qpos[1])
            data_dict['q1_dot'].append(prev_qvel[0])
            data_dict['q2_dot'].append(prev_qvel[1])
            data_dict['q1_ddot'].append(qacc[0])
            data_dict['q2_ddot'].append(qacc[1])
            data_dict['tau1'].append(torque[0])
            data_dict['tau2'].append(torque[1])
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()

    for t in range(10000):
        torque1 = math.sin(t * 0.05)
        torque2 = math.cos(t * 0.05)
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()

    for t in range(10000):
        torque1 = math.cos(t * 0.05)
        torque2 = math.sin(t * 0.05)
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()
    
    for t in range(5000):
        torque1 = 1.0
        torque2 = 1.0
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()
    
    for t in range(1000):
        torque1 = -1.0
        torque2 = -1.0
        action = np.array([torque1, torque2], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)

        qpos = env.unwrapped.data.qpos[:2].copy()
        qvel = env.unwrapped.data.qvel[:2].copy()
        torque = env.unwrapped.data.ctrl[:2].copy()

        if prev_qvel is not None:
            # calculate acceleration
            qacc = (qvel - prev_qvel) / dt
            data.append(np.hstack([qpos, prev_qvel, qacc, torque]))

        prev_qvel = qvel.copy()

    env.close()
    # Save to CSV
    with open("reacher_samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q1", "q2", "q1_dot", "q2_dot", "q1_ddot", "q2_ddot", "tau1", "tau2"])
        writer.writerows(data)

def regressor():
    df = pd.read_csv("reacher_samples.csv")
    Y, tau = [], []

    for _, row in df.iterrows():
        q1 = row.q1
        q2 = row.q2
        q1_dot = row.q1_dot
        q2_dot = row.q2_dot
        q1_ddot = row.q1_ddot
        q2_ddot = row.q2_ddot

        # Regressor row for tau1
        y1 = [
            q1_ddot,
            np.cos(q2)*(2 * q1_ddot + q2_ddot) - np.sin(q2)* q2_dot * (2 * q1_dot + q2_dot),
            q2_ddot
        ]

        # Regressor row for tau2
        y2 = [
            0.0,
            np.cos(q2) * q1_ddot + np.sin(q2) * q1_dot**2,
            q1_ddot + q2_ddot
        ]

        Y.extend([y1, y2])
        tau.extend([row.tau1, row.tau2])

    Y = np.array(Y)
    tau = np.array(tau)

    np.set_printoptions(suppress=True)
    
    # # Estimate parameters
    # p, _, _, _ = np.linalg.lstsq(Y, tau, rcond=None)
    # results = pd.DataFrame({
    #     'parameter': ['alpha', 'beta', 'gamma'],
    #     'value': p,
    # })
    
    # print("\nEstimated Parameters:")
    # print(results)
    # print(p)

    Y = np.linalg.pinv(Y)
    
    # Estimate parameters
    p = Y @ tau
    results1 = pd.DataFrame({
        'parameter': ['alpha', 'beta', 'gamma'],
        'value': p,
    })
    print("\nEstimated Parameters:")
    print(results1)
    print(p)

        
def main():
    get_samples()
    regressor()

if __name__ == "__main__":
    main()
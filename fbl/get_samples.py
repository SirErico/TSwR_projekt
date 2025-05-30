import gymnasium as gym
import numpy as np
import pandas as pd
import csv

def get_samples():
    env = gym.make("Reacher-v5", render_mode="none")
    obs, _ = env.reset(seed=0)

    data_dict = {
        'q1': [], 'q2': [], 
        'q1_dot': [], 'q2_dot': [], 
        'q1_ddot': [], 'q2_ddot': [], 
        'tau1': [], 'tau2': []
    }
    
    prev_qvel = None
    dt = env.unwrapped.model.opt.timestep

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
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

        prev_qvel = qvel.copy()

    env.close()

    df = pd.DataFrame(data_dict)
    df.to_csv("reacher_dynamics_samples.csv", index=False)
    return df

def regressor(df: pd.DataFrame):
    Y, tau_list = [], []

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
            2 * np.cos(q2) * q1_ddot - 2 * np.sin(q2) * q1_dot * q2_dot + np.cos(q2)*q2_ddot - np.sin(q2) * q2_dot**2,
            q2_ddot
        ]

        # Regressor row for tau2
        y2 = [
            0.0, 
            np.cos(q2) * q1_ddot + np.sin(q2) * q1_dot**2,
            q1_ddot + q2_ddot  
        ]

        Y.extend([y1, y2])
        tau_list.extend([row.tau1, row.tau2])

    Y = np.array(Y)
    tau = np.array(tau_list)
    
    # Estimate parameters
    p, _, _, _ = np.linalg.lstsq(Y, tau, rcond=None)
    results = pd.DataFrame({
        'parameter': ['alpha', 'beta', 'gamma'],
        'value': p,
    })
    
    print("\nEstimated Parameters:")
    print(results)
    return results
        
def main():
    df = get_samples()
    print("\nSample Statistics:")
    print(df.describe())
    
    results = regressor(df)

if __name__ == "__main__":
    main()
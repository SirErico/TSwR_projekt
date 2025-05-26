# There are many different RL Algorithms...

<p align="center">
  <img src="https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg" width="400">
</p>

We knew we had to look for algorithms that support a continuous action space and work model-free.

## Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy algorithm that:
- Combines DQN and policy gradient methods
- Uses deterministic policies
- Employs replay buffer and target networks

Key components:
- Actor-critic architecture
- Ornstein-Uhlenbeck noise for exploration
- Soft target updates
- Experience replay


## Proximal Policy Optimization (PPO)

PPO is an on-policy algorithm that:
- Uses clipped surrogate objective function to prevent too large policy updates
- Alternates between sampling data and optimization
- Is relatively simple to implement and tune


## Soft Actor-Critic (SAC)

SAC is an off-policy algorithm that:
- Maximizes both expected return and entropy
- Uses soft Q-learning and double Q-networks
- Performs well in exploration-heavy tasks

Key features:
- Automatic entropy tuning
- Twin critics for value estimation
- Separate policy and value networks
- Replay buffer for off-policy learning


## Twin Delayed Deep Deterministic Policy Gradient (TD3)

TD3 is an off-policy algorithm that improves upon DDPG by:
- Using twin critics to reduce overestimation bias
- Delaying policy updates
- Adding noise to target actions
- Clipping noise in target policy

Key improvements:
- Twin critics (reduces overestimation)
- Delayed policy updates (more stable learning)
- Target policy smoothing (better generalization)


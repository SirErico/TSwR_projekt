# What do we know about the Reacher:


### 1. Action space 
The action space is Continuous. It's a Box(-1.0, 1.0, (2,), float32). 
- `action[0]` is the torque at the first joint (connects the base with the first link)
- `action[1]` is the torque at the second joint (between first and second link)

---

### 2. Observation space
The observation space is an array with shape (11,), where the elemenets correcpond to:

|Num|Observation|Min|Max|Name (in corresponding XML file)|Joint|Unit|
|---|---|---|---|---|---|---|
|0|cosine of the angle of the first arm|-Inf|Inf|cos(joint0)|hinge|unitless|
|1|cosine of the angle of the second arm|-Inf|Inf|cos(joint1)|hinge|unitless|
|2|sine of the angle of the first arm|-Inf|Inf|cos(joint0)|hinge|unitless|
|3|sine of the angle of the second arm|-Inf|Inf|cos(joint1)|hinge|unitless|
|4|x-coordinate of the target|-Inf|Inf|target_x|slide|position (m)|
|5|y-coordinate of the target|-Inf|Inf|target_y|slide|position (m)|
|6|angular velocity of the first arm|-Inf|Inf|joint0|hinge|angular velocity (rad/s)|
|7|angular velocity of the second arm|-Inf|Inf|joint1|hinge|angular velocity (rad/s)|
|8|x-value of position_fingertip - position_target|-Inf|Inf|NA|slide|position (m)|
|9|y-value of position_fingertip - position_target|-Inf|Inf|NA|slide|position (m)|
|10|z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both)|-Inf|Inf|NA|slide|position (m)|

---

### 3. Reward function
The reward is composed of two parts:
- 'reward_distance': this reward is a measure of the distance between the end of the reacher and the target, with more negative values assigned for when the reacher is further away from target.
- 'reward_control': A small penalty(negative reward) for using large torques (action too large).
The formula used:
```python
reward = reward_distance + reward_control
```
The agent gets rewarded for minimizing the distance efficiently.

#### Episode end:
- Truncation: the episode duration reached 50 timesteps
- Termination: any of the state space values is no longer finite
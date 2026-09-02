# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The objective is to train an agent to learn the optimal action-value function and find a suitable path from the starting state to the goal while avoiding holes. The agent should use an epsilon-greedy strategy to balance exploration and exploitation.



## Software Requirements
Programming Language: Python 3
Platform: Google Colab / Jupyter Notebook
Libraries:
Gymnasium – for the FrozenLake environment
NumPy – for Q-table and numerical calculations
Matplotlib – for plotting the learning curve
Algorithm: Q-Learning
Environment: FrozenLake-v1
Hardware: Basic computer with internet access
Python packages:



## Environment Description

FrozenLake-v1 is a grid-world reinforcement learning environment consisting of a 4 × 4 grid. The agent starts from the initial state and must reach the goal while avoiding holes.

States: 16
Actions: 4
0 – Left
1 – Down
2 – Right
3 – Up
Reward:
+1 for reaching the goal
0 for other transitions, including falling into a hole
Episode termination: The episode ends when the agent reaches the goal or falls into a hole.
Slippery environment: is_slippery=True, so the agent's movement can be stochastic.
Learning: The Q-table is updated using the Q-Learning update rule, while epsilon-greedy action selection allows the agent to explore and exploit.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1. Initialize the FrozenLake environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros.
4. Set the learning rate `α`.
5. Set the discount factor `γ`.
6. Initialize epsilon `ε`.
7. Repeat for the specified number of episodes:
   * Reset the environment.
   * Obtain the initial state.
   * Select an action using epsilon-greedy action selection.
   * Execute the action.
   * Observe the next state and reward.
   * Update the Q-value using the Q-Learning equation.
   * Move to the next state.
   * Continue until the episode terminates.
8. Decrease epsilon after each episode.
9. Calculate the estimated state-value function.
10. Extract the learned policy using the maximum Q-value for each state.
11. Calculate the average reward over the final 1000 episodes.
12. Plot the training performance.


## Python Program

```python
#-------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(num_episodes):

    state, info = env.reset()
    total_reward = 0

    done = False

    while not done:

        # Choose action using epsilon-greedy
        action = choose_action(state, epsilon)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Q-Learning update
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Extract State-Value Function and Policy
# -------------------------------------------------

# V(s) = max_a Q(s,a)
state_values = np.max(Q, axis=1)

# π(s) = argmax_a Q(s,a)
learned_policy = np.argmax(Q, axis=1)



```

## Output

#### Final Q-table:
<img width="463" height="422" alt="image" src="https://github.com/user-attachments/assets/9f57d928-f755-4f7e-847d-d262542e3e6f" />

#### Estimated State-Value Function:
 <img width="881" height="152" alt="image" src="https://github.com/user-attachments/assets/31e92a96-330d-46d3-ba91-6f9e31ad57af" />


#### Learned Policy:
<img width="765" height="143" alt="image" src="https://github.com/user-attachments/assets/fa9fe674-0480-4992-af72-b9f5a6fa5b44" />


#### Average reward over last 1000 episodes: 
<img width="695" height="35" alt="image" src="https://github.com/user-attachments/assets/aa61a375-a077-4b80-8b67-d9ab802d92f1" />



#### Q-Learning Curve - FrozenLake:
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/2e1dfdb3-5412-4ffe-9f13-9d5e64653795" />






## Result
Q-Learning successfully learned a policy for reaching the goal in FrozenLake while avoiding holes.







---

## Inference

The experiment demonstrates that Q-Learning can learn an effective policy through trial and error without being given the optimal path beforehand. By balancing exploration and exploitation using epsilon-greedy action selection, the agent gradually learns better actions for reaching the goal while avoiding holes. Thus, Q-Learning is effective for solving the FrozenLake reinforcement learning problem.






---


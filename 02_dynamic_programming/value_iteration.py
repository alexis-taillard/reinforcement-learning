import numpy as np
import matplotlib.pyplot as plt

import os
os.makedirs("../figures", exist_ok=True)

states = 8 
A = ['l', 'r', 'b']  # we add "b" representing the back action
actions = 3

# Rewards
Reward_A = [[0, 0, 0], [2, 2, 2], [1, 1, 1], [-1, -1, -1], [3, 3, 3], [-3, -3, -3], [-7, -7, -7], [5, 5, 5]]    # we have 3 rewards per states for all the actions
Reward_B = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 5, 5]] #  Sparse reward : From what I understood, a sparse reward means the agent only gets a reward
# at the very end (here +5 in S8, the terminal state) and 0 everywhere else.

 #[Left, Right, Back]
TransitionProbability = [
    #   S1          S2            S3            S4            S5            S6            S7            S8
    [[0.0,0.0,0.0],[0.7,0.3,0.0],[0.3,0.7,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],  # S1
    [[0.0,0.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.7,0.3,0.0],[0.3,0.7,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],  # S2
    [[0.0,0.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.7,0.3,0.0],[0.3,0.7,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],  # S3
    [[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],  # S4
    [[0.0,0.0,0.0],[0.0,0.0,0.5],[0.0,0.0,0.5],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.7,0.3,0.0],[0.3,0.7,0.0]],  # S5    from S5 we can go back to S2 and S3 so I decide to split probability (0.5 , 0.5)
    [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[1.0,1.0,0.0]],  # S6
    [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],  # S7
    [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]   # S8
]

# S8 only terminal state
Terminal = ['', '', '', '', '', '', '', 'T']
Value = [0, 0, 0, 0, 0, 0, 0, 0]  # Initial Value estimation of each state (we can set random too)
gamma = 0.9
delta = 0.001

# Part A

for iteration in range(0, 100):
    NewValue = [-1e10] * states
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward_A[j][a] + gamma * Value[j])
            NewValue[i] = round(max(NewValue[i], value_temp), 5)
    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i]-NewValue[i]))
    Value = NewValue
    print(iteration, NewValue, 'bellman_factor (' + str(bellman_factor) + ')' , sep=",    ")
    if(bellman_factor < delta):
        break

# Determine the policy (One time iteration)
NewValue = [-1e10] * states
policy = ['NA','NA','NA','NA','NA', 'NA', 'NA', 'NA']
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (Reward_A[j][a] + gamma * Value[j])
        if(NewValue[i] < value_temp):
            if(Terminal[i] != 'T'):
                policy[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy[i] = 'T'

print("The algorithm's final policy (part A) is:", policy)

# Part B

print("\n PART B : SPARSE REWARD")
Value_B = [0, 0, 0, 0, 0, 0, 0, 0]
hist_sum_B = []
hist_delta_B = []

for iteration in range(0, 100):
    NewValue = [-1e10] * states
    for i in range(states):
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (
                    Reward_B[j][a] + gamma * Value_B[j]
                )
            NewValue[i] = round(max(NewValue[i], value_temp), 5)

    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value_B[i] - NewValue[i]))

    Value_B = NewValue
    hist_sum_B.append(sum(Value_B))
    hist_delta_B.append(bellman_factor)

    print(
        iteration,
        NewValue,
        "bellman_factor (" + str(bellman_factor) + ")",
        sep=",    ",
    )
    if bellman_factor < delta:
        break

NewValue = [-1e10] * states
policy_B = ["NA"] * states
for i in range(states):
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (
                Reward_B[j][a] + gamma * Value_B[j]
            )
        if NewValue[i] < value_temp:
            if Terminal[i] != "T":
                policy_B[i] = A[a]
                NewValue[i] = max(NewValue[i], value_temp)
            else:
                policy_B[i] = "T"

print("Final policy Part B:", policy_B)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(range(len(hist_sum_B)), hist_sum_B, marker='o')
ax[0].set_title("Part B")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("sum of V(s)")
ax[0].grid(True)

ax[1].semilogy(range(len(hist_delta_B)), hist_delta_B, marker='o')  # logaritmic scale
ax[1].axhline(delta, color='r', linestyle='--', label=f"threshold δ = {delta}")
ax[1].set_title("Part B : bellman_factor")
ax[1].set_xlabel("Iteration")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig("../figures/vi_sparse_reward.png", dpi=150, bbox_inches="tight")

# Analysis

# Because intermediate states provide positive rewards, introducing the back action allows the agent to exploit infinite loops to maximize its long-term return rather than reaching the terminal state S8.
# For Part B, left plot : The sum of state values grows rapidly before reaching a steady plateau around iteration 10–12 as the single +5 reward from S8 backpropagates across all states.
#  The maximum update per iteration decreases geometrically until it falls below delta = 0.001 at iteration 15, confirming convergence and stopping the algorithm.
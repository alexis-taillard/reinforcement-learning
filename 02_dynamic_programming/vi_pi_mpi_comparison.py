import numpy as np
import matplotlib.pyplot as plt

import os
os.makedirs("../figures", exist_ok=True)

states = 9
A = ['l', 'r']
actions = 2

Reward = [[0,0],[2,0],[0,1],[3,0],[-2,-2],[0,-3],[0,-5],[0,3],[5,5]]

# TransitionProbability[i][j][a] = P(j | i, a)
TransitionProbability = [
    [[0.0,0.0],[0.8,0.2],[0.2,0.8],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.8,0.2],[0.2,0.8],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.8,0.2],[0.2,0.8],[0.0,0.0],[0.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,1.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,1.0],[0.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[1.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[1.0,0.0]],
    [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]],
]

Terminal = ['','','','','','','T','','T']

gamma = 0.9
delta = 0.001


# ==========================================
# 1. VALUE ITERATION

Value = [0, 0, 0, 0, 0, 0, 0, 0, 0]
hist_VI = [list(Value)]   # for convergence graph

print("\n=== VALUE ITERATION ===")

for iteration in range(0, 100):
    NewValue = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(states):
        if Terminal[i] == 'T':
            continue
        max_v = -1e10
        for a in range(actions):
            value_temp = 0
            for j in range(states):
                value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
            max_v = max(max_v, value_temp)
        NewValue[i] = round(max_v, 5)

    bellman_factor = 0
    for i in range(states):
        bellman_factor = max(bellman_factor, abs(Value[i] - NewValue[i]))
    
    Value = NewValue
    hist_VI.append(list(Value))
    print(iteration, NewValue, 'bellman_factor (' + str(round(bellman_factor, 5)) + ')', sep=",    ")
    if bellman_factor < delta:
        break

# Determine the policy for VI
policy = ['NA'] * states
for i in range(states):
    if Terminal[i] == 'T':
        policy[i] = 'T'
        continue
    best_v = -1e10
    for a in range(actions):
        value_temp = 0
        for j in range(states):
            value_temp += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value[j])
        if value_temp > best_v:
            best_v = value_temp
            policy[i] = A[a]

print("The algorithm's final policy is:", policy)


# ==========================================
# 2. POLICY ITERATION

Value2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
policy2 = ['l', 'l', 'l', 'l', 'l', 'l', 'T', 'l', 'T']
hist_PI = [list(Value2)]

print("\n=== POLICY ITERATION ===")

for policy_iteration in range(0, 100):
    
    # 1. Policy Evaluation
    for eval_iteration in range(0, 100):
        NewValue2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(states):
            if Terminal[i] == 'T':
                continue
            a = A.index(policy2[i])
            value_temp2 = 0
            for j in range(states):
                value_temp2 += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value2[j])
            NewValue2[i] = round(value_temp2, 5)

        bellman_factor2 = 0
        for i in range(states):
            bellman_factor2 = max(bellman_factor2, abs(Value2[i] - NewValue2[i]))
        Value2 = NewValue2

        if bellman_factor2 < delta:
            break

    # 2. Policy Improvement
    policy_stable = True
    for i in range(states):
        if Terminal[i] == 'T':
            continue
        
        old_action = policy2[i]
        best_value = -1e10
        best_action = policy2[i]

        for a in range(actions):
            value_temp2 = 0
            for j in range(states):
                value_temp2 += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value2[j])
            if value_temp2 > best_value:
                best_value = value_temp2
                best_action = A[a]

        policy2[i] = best_action
        if old_action != best_action:
            policy_stable = False

    hist_PI.append(list(Value2))
    print(f"Policy Iteration {policy_iteration}: Policy = {policy2}, Values = {Value2}")

    if policy_stable:
        break

print("Policy Iteration final policy is:", policy2)


# 3. MODIFIED POLICY ITERATION

Value3 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
policy3 = ['l', 'l', 'l', 'l', 'l', 'l', 'T', 'l', 'T']
k_eval_steps = 3
hist_MPI = [list(Value3)]

print("\n=== MODIFIED POLICY ITERATION ===")

for mpi_iteration in range(0, 100):
    
    # 1. PARTIAL POLICY EVALUATION
    for _ in range(k_eval_steps):
        NewValue3 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(states):
            if Terminal[i] == 'T':
                continue
            a = A.index(policy3[i])
            value_temp3 = 0
            for j in range(states):
                value_temp3 += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value3[j])
            NewValue3[i] = round(value_temp3, 5)

        Value3 = NewValue3

    # 2. POLICY IMPROVEMENT
    policy_stable = True
    bellman_factor3 = 0
    
    for i in range(states):
        if Terminal[i] == 'T':
            continue
        
        old_action = policy3[i]
        best_value = -1e10
        best_action = policy3[i]

        for a in range(actions):
            value_temp3 = 0
            for j in range(states):
                value_temp3 += TransitionProbability[i][j][a] * (Reward[j][a] + gamma * Value3[j])
            
            if value_temp3 > best_value:
                best_value = value_temp3
                best_action = A[a]

        bellman_factor3 = max(bellman_factor3, abs(Value3[i] - best_value))
        policy3[i] = best_action
        
        if old_action != best_action:
            policy_stable = False

    hist_MPI.append(list(Value3))
    print(f"Iteration {mpi_iteration}: Policy = {policy3}, Values = {Value3}, bellman_factor = {round(bellman_factor3, 5)}")

    if policy_stable and bellman_factor3 < delta:
        break

print("Modified Policy Iteration final policy is:", policy3)


V_star = np.array(Value)

def erreur(hist):

    return [np.max(np.abs(np.array(v) - V_star)) for v in hist]

plt.figure(figsize=(8, 5))
plt.plot(erreur(hist_VI),  'o-', label='Value Iteration')
plt.plot(erreur(hist_PI),  's-', label='Policy Iteration')
plt.plot(erreur(hist_MPI), '^-', label=f'Modified Policy Iteration (k={k_eval_steps})')
plt.xlabel('Iteration')
plt.ylabel('max |V - V*|')
plt.title('Convergence speed')
plt.legend()
plt.tight_layout()
plt.savefig("../figures/vi_pi_mpi_convergence.png", dpi=150, bbox_inches="tight")

# policy iteration and modified converges with 2 iterations for this problem and value iteration with 5 
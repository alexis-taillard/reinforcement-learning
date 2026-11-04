import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import os
os.makedirs("../figures", exist_ok=True)

np.random.seed(20) # Numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness.

Number_of_Bandits = 4
p_bandits = [0.5, 0.1, 0.8, 0.3]
N = 1000                          # number of steps (iterations)

def bandit_run(index):
    if np.random.rand() >= p_bandits[index]: # random  probability to win or lose per machine
        return -1 # Lose
    else:
        return 1 # Win


def total_win_rate(win_count, running_count):
    total = 0
    for i in range(Number_of_Bandits):
        total += win_count[i] / running_count[i]

    return total


# 1. EPSILON-GREEDY

def run_epsilon_greedy(N):
    epsilon = 0.1       # 10% exploration and 90% exploitation, a good balance between exploring and exploiting
    start_time = time.time()

    counts = np.zeros(Number_of_Bandits)     # initialisation of how many times each arm was played
    values = np.zeros(Number_of_Bandits)     # our current estimate Q(a) for each arm

    win_count = [0] * Number_of_Bandits
    running_count = [1] * Number_of_Bandits    # Number of runs for each arm

    reward_curve = np.zeros(N)      # list to store current reward at each iteration

    for t in range(N):
        # Explore randomly or exploit the best arm
        if np.random.rand() < epsilon:
            action = np.random.choice(Number_of_Bandits)
        else:
            action = np.argmax(values)  # Choose the arm with the highest value

        reward = bandit_run(action)
        counts[action] += 1
        # Update the estimated average reward
        values[action] += (reward - values[action]) / counts[action]

        if reward == 1:
            win_count[action] += 1
        running_count[action] += 1
        reward_curve[t] = total_win_rate(win_count, running_count)  # Calculate the current win rate to put it in the plot later

    execution_time = time.time() - start_time   # Calculate execution time of the algorithm
    total_reward = sum(win_count)
    return reward_curve, execution_time, total_reward


#2. EPSILON-GREEDY (epsilon decreasing)

def run_epsilon_greedy_decay(N):
    epsilon = 0.1
    epsilon_min = 0.01  # Prevent exploration from becoming too low after epsilon decay
    decay = 0.995

    start_time = time.time()

    counts = np.zeros(Number_of_Bandits)
    values = np.zeros(Number_of_Bandits)

    win_count = [0] * Number_of_Bandits
    running_count = [1] * Number_of_Bandits

    reward_curve = np.zeros(N)

    for t in range(N):
        if np.random.rand() < epsilon:
            action = np.random.choice(Number_of_Bandits)
        else:
            action = np.argmax(values)

        reward = bandit_run(action)
        counts[action] += 1

        values[action] += (reward - values[action]) / counts[action]

        if reward == 1:
            win_count[action] += 1

        running_count[action] += 1
        reward_curve[t] = total_win_rate(win_count, running_count)

        # Decrease epsilon over time
        epsilon = max(epsilon_min, epsilon * decay)     # I am taking max here in order to not get below epsilon_min

    execution_time = time.time() - start_time
    total_reward = sum(win_count)

    return reward_curve, execution_time, total_reward           # returning reward's evolution during time, execution time and total reward


# 3. UCB (Upper Confidence Bounds)

def run_ucb(N):
    start_time = time.time()

    counts = np.zeros(Number_of_Bandits)
    values = np.zeros(Number_of_Bandits)

    win_count = [0] * Number_of_Bandits
    running_count = [1] * Number_of_Bandits

    reward_curve = np.zeros(N)

    for t in range(N):
        if t < Number_of_Bandits:   # therefore we play each arm once at the beginning
            action = t
        else:
            ucb_values = values + np.sqrt((2 * np.log(t + 1)) / counts)           # We Calculate the bound value for each arm
            action = np.argmax(ucb_values)

        reward = bandit_run(action)
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]

        if reward == 1:
            win_count[action] += 1
        running_count[action] += 1
        reward_curve[t] = total_win_rate(win_count, running_count)

    execution_time = time.time() - start_time
    total_reward = sum(win_count)
    return reward_curve, execution_time, total_reward


# 4. THOMPSON SAMPLING

def run_thompson_sampling(N):
    start_time = time.time()

    wins = np.zeros(Number_of_Bandits)
    losses = np.zeros(Number_of_Bandits)

    win_count = [0] * Number_of_Bandits
    running_count = [1] * Number_of_Bandits

    reward_curve = np.zeros(N)

    for t in range(N):
        # sample a random value for each arm from its Beta distribution
        samples = []        # creating a list where we are going to put the result for each arm
        for i in range(Number_of_Bandits):
            samples.append(stats.beta.rvs(wins[i] + 1, losses[i] + 1))

        action = np.argmax(samples)
        reward = bandit_run(action)

        if reward == 1:
            wins[action] += 1
        else:
            losses[action] += 1

        if reward == 1:
            win_count[action] += 1
        running_count[action] += 1
        reward_curve[t] = total_win_rate(win_count, running_count)

    execution_time = time.time() - start_time
    total_reward = sum(win_count)
    return reward_curve, execution_time, total_reward


np.random.seed(20)
rewards_eps, time_eps, total_eps = run_epsilon_greedy(N)        # running the epsilon-greedy algorithm

np.random.seed(20)
rewards_decay, time_decay, total_decay = run_epsilon_greedy_decay(N)

np.random.seed(20)
rewards_ucb, time_ucb, total_ucb = run_ucb(N)               # running the ucb algo

np.random.seed(20)
rewards_ts, time_ts, total_ts = run_thompson_sampling(N)        # running the thompson algo

# printing number of wins and time used for each algorithm

print(f"--- Results over {N} steps ---")
print(f"Epsilon-Greedy       : {total_eps} victoires, {time_eps*1000:.2f} ms")
print(f"Epsilon-Greedy Decay : {total_decay} victoires, {time_decay*1000:.2f} ms")
print(f"UCB                  : {total_ucb} victoires, {time_ucb*1000:.2f} ms")
print(f"Thompson Sampling    : {total_ts} victoires, {time_ts*1000:.2f} ms")


# PLOT 1 - reward comparison

plt.figure(figsize=(9, 6))
plt.title('Sum of Win Rates Comparison')
plt.plot(rewards_eps, color='blue', label='Epsilon-Greedy')
plt.plot(rewards_decay, color ='pink', label='Epsilon-Greedy Decay')
plt.plot(rewards_ucb, color='orange', label='UCB')
plt.plot(rewards_ts, color='green', label='Thompson')
plt.xlabel('Episode')
plt.ylabel('Sum of win rates')
plt.grid(axis='x', color='0.80')
plt.legend(title='Algorithm :')
plt.tight_layout()
plt.savefig("../figures/mab_win_rates.png", dpi=150, bbox_inches="tight")


# PLOT 2 - execution time comparison

plt.figure(figsize=(7, 6))      # new window
algo_names = ['Epsilon-Greedy', 'Epsilon-Greedy Decay', 'UCB', 'Thompson']
times_ms = [time_eps * 1000, time_decay * 1000, time_ucb * 1000, time_ts * 1000]        #  *1000 to get times in ms
bar_colors = ['blue', 'pink', 'orange', 'green']

bars = plt.bar(algo_names, times_ms, color=bar_colors)
plt.title('Execution Time Comparison')
plt.ylabel('Time (ms)')
plt.grid(axis='y', linestyle='--', alpha=0.6)       # grid only on the y axis

for bar, t in zip(bars, times_ms):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.1f} ms",                 # Display the exact value (in ms) above each bar, centered and bold
              ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("../figures/mab_execution_time.png", dpi=150, bbox_inches="tight")




# Comparison:
# Thompson Sampling achieves the highest sum of win rates, around 1.75, while UCB reaches around 1.62 and Epsilon-Greedy around 1.50.
# However, Thompson Sampling is much slower, with an execution time of about 182.6 ms, compared to 9.3 ms for UCB and 7.5 ms for Epsilon-Greedy.
# This is because Thompson Sampling generates a Beta-distributed sample for every arm at each iteration, which requires more computations.
# Epsilon-Greedy with decay achieves a higher sum of win rates than without decay.
# Epsilon-Greedy decay improves the performance of Epsilon-Greedy while keeping a similar computational cost.

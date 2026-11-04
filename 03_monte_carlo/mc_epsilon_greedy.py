# Problem: A robot needs to go from Start to Target
# Reward: +100 for step into the target, -1 for each other step (transition-based reward)
# Q-value: for simplicity all zero
# Initial Q values are all zero
# Extra actions: jump S19 -> S9 and jump S20 -> S2

import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.makedirs("../figures", exist_ok=True)

np.random.seed(9527)
random.seed(9527)

class Environment():
    def __init__(self):
        self.rows = 4
        self.cols = 6
        self.grid_world = [[  "T",  "s1",  "s2",  "s3",  "s4",  "s5"],
                           [ "s6",  "s7",  "s8",  "s9",   "W", "s10"],
                           ["s11",   "W", "s12",   "W", "s13", "s14"],
                           ["s15", "s16", "s17", "s18", "s19", "s20"]] # T: Target, W: Wall

        self.action_to_number = {"up": 0, "right": 1, "down": 2, "left": 3, "jump": 4}
        self.action_dict = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1), "jump": (0, 0)}
        self.direction_dict = {0: "up", 1: "right", 2: "down", 3: "left", 4: "jump"}
        self.invalid_start = ["T", "W"]

        # The jumps only exist from S19 and S20
        self.jumps = {(3, 4): (1, 3),   # S19 -> S9
                      (3, 5): (0, 2)}   # S20 -> S2
        self.start = (3, 5)             # S20

    def step(self, state_coordinates, action): # Input(state, action), Output(next state)
        current_state_coordinates = np.array(state_coordinates)
        next_state_coordinates = current_state_coordinates

        if action == "jump":
            target = self.jumps.get(tuple(int(x) for x in current_state_coordinates)) # only S19 or S20 can jump
            if target is not None:
                next_state_coordinates = np.array(target)  # jump to the destination state
            # elsewhere: nothing happens (the agent stays in the same state)
        else:
            new_coords = current_state_coordinates + self.action_dict[action]

            if (0 <= new_coords[0] <= 3 and 0 <= new_coords[1] <= 5):  # inside the board
                if self.grid_world[new_coords[0]][new_coords[1]] != "W":  # not a wall
                    next_state_coordinates = new_coords

        return tuple(int(x) for x in next_state_coordinates)


class Monte_Carlo_Epsilon_Greedy():
    def __init__(self):
        self.env = Environment()
        self.Max_iteration = 10000
        self.epsilon = 0.1
        self.gamma = 0.9
        self.Horizon = 10   # Max episode_length

        self.Q_values = {}
        self.returns_dict = {}
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] not in self.env.invalid_start:
                    for act in self.env.action_dict.keys():
                        self.Q_values[((row, col), act)] = 0       # Initialize Q value (state, action)
                        self.returns_dict[((row, col), act)] = [0, 0]  # [Mean value, Visited count]

        self.episode_rewards = []

    def policy(self, state):
        valid_actions = []
        for action in self.env.action_dict.keys():
            if self.env.step(state, action) != state:
                valid_actions.append(action)

        Q_value = [self.Q_values[(state, a)] for a in valid_actions]
        max_value = max(Q_value)

        indexes = []
        for action in valid_actions:
            if max_value == self.Q_values[(state, action)]:
                indexes.append(self.env.action_to_number[action])
        return self.env.direction_dict[random.choice(indexes)]

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:          # condtiion of the epsilon-greedy algorithm : in 10% of time, we explore <ith a random move
            action = self.env.direction_dict[random.randint(0, 4)]
            while self.env.step(state, action) == state:
                action = self.env.direction_dict[random.randint(0, 4)]
            return action                                   # explore
        return self.policy(state)                # exploit

    def print_Qvalue(self, state):
        Q_value = []
        for action in self.env.action_dict.keys():
            Q_value.append(self.Q_values[(state, action)])
        return Q_value

    def iterate(self): #Main loop
        for iteration in range(self.Max_iteration):
            episode = []
            self.current_state_coordinates = self.env.start   # always starting from s20 and not randomly
            action = self.epsilon_greedy_action(self.current_state_coordinates)
            for h in range(self.Horizon): #Generate episode
                next_state_coordinates = self.env.step(self.current_state_coordinates, action)
                reward = -1
                if self.env.grid_world[next_state_coordinates[0]][next_state_coordinates[1]] == "T":
                    reward = 100
                    episode.append([self.current_state_coordinates, action, reward])
                    break
                episode.append([self.current_state_coordinates, action, reward]) #Episode [[[coordinate],action, reward]]
                self.current_state_coordinates = next_state_coordinates
                action = self.epsilon_greedy_action(self.current_state_coordinates)
            self.episode_rewards.append(sum(step[2] for step in episode))  # for the plot
            G = 0

            for h in range(len(episode)-1, -1, -1): #Iterate through H-1, H-2,...,0
                coordinate, action, reward = episode[h]
                G = self.gamma*G + reward
                returns = self.returns_dict[(tuple(coordinate), action)]
                mean = returns[0]
                visited_count = returns[1]
                mean = (mean*visited_count + G)/(visited_count + 1)
                visited_count += 1
                self.returns_dict[(tuple(episode[h][0]), episode[h][1])] = [mean, visited_count]
                self.Q_values[(tuple(episode[h][0]), episode[h][1])] = self.returns_dict[(tuple(episode[h][0]), episode[h][1])][0]

    def render(self):
        output = [row[:] for row in self.env.grid_world]
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if self.env.grid_world[row][col] in self.env.invalid_start:
                    continue
                else:
                    output[row][col] = self.policy((row, col))

        for row in range(0, self.env.rows):
            print("-------------------------------------------------------")
            out = "| "
            for col in range(0, self.env.cols):
                out += str(output[row][col]).ljust(6) + " | "
            print(out)
        print("-------------------------------------------------------")

    def follow_greedy_policy(self):  # Path of the final policy from the start state which is s20
        state = self.env.start
        path, total, done = [], 0, False
        for h in range(self.Horizon):
            action = self.policy(state)
            next_state = self.env.step(state, action)
            reward = -1
            if self.env.grid_world[next_state[0]][next_state[1]] == "T":
                reward = 100
                done = True
            path.append((state, action))
            total += reward
            state = next_state
            if done:
                break
        return path, total, done

    def plot(self):
        rewards = np.array(self.episode_rewards)    # getting all rewards for the plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        axes[0].plot(np.cumsum(rewards))
        axes[0].set_title("Accumulated total reward")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cumulative reward")
        axes[0].grid(True)

        axes[1].plot(rewards, label="reward per episode")
        axes[1].set_title("Total reward per episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle(f"MC epsilon-greedy (epsilon={self.epsilon}, gamma={self.gamma}, "
                     f"horizon={self.Horizon}, {self.Max_iteration} episodes)")
        plt.savefig("../figures/mc_epsilon_greedy.png", dpi=150, bbox_inches="tight")       

if __name__ == "__main__":
    MC = Monte_Carlo_Epsilon_Greedy()
    print("Before learning (all Q = 0), T = Target, W = Wall")
    MC.render()
    MC.iterate()
    print("\nLearned greedy policy, T = Target, W = Wall")
    MC.render()

    path, total, reached = MC.follow_greedy_policy()
    print("\nGreedy path from S20:", path)
    print("Target reached:", reached, "| total reward:", total)
    print("Mean reward over the last 500 episodes:", np.mean(MC.episode_rewards[-500:]))
    MC.plot()

### With the test from s20 to s2, the agent jumps and then goes twice left to reach the final state s0, so the jump implementation seems working
# with the right plot, we see some episodes taking a negative amount of rewards sometimes, it is when the agent is exploring and not reaching the final state with the big reward
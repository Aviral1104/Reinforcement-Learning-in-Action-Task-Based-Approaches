'''
below mentioned is the custom grid for which the agent is trained
    0    1    2    3    4
0 [ 0. | 0. | 0. | 0. | 0. ]
1 [ S  | 0. | 0. | J  | 0. ]
2 [ 0. | 0. | X  | 0. | 0. ]
3 [ 0. | 0. | X  | X  | 0. ]
4 [ 0. | 0. | X  | 0. | G  ]

This environment represents a navigation task where an agent must move from the start to the goal while avoiding obstacles. 
The jump position offers a bonus reward. The agent receives positive rewards for reaching the goal or jump position, negative rewards for obstacles, and zero rewards for empty cells.
'''

import numpy as np
import gym
from gym import spaces
import random

class GridWorldEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(25)
        self.action_space = spaces.Discrete(4)
        self.grid_size = 5
        self.start_position = (1, 0)
        self.goal_position = (4, 4)
        self.jump_position = (1, 3)
        self.obstacle_positions = [(2, 2), (3, 2), (3, 3), (4, 2)]
        self.current_position = self.start_position
        self.reward_range = (-np.inf, np.inf)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[self.goal_position[0], self.goal_position[1]] = 10
        self.grid[self.jump_position[0], self.jump_position[1]] = 5
        for obstacle_pos in self.obstacle_positions:
            self.grid[obstacle_pos[0], obstacle_pos[1]] = -1

    def reset(self):
        self.current_position = self.start_position
        return self.current_position[0] * self.grid_size + self.current_position[1]

    def step(self, action):
        if action == 0:  # North
            next_position = (self.current_position[0] - 1, self.current_position[1])
        elif action == 1:  # South
            next_position = (self.current_position[0] + 1, self.current_position[1])
        elif action == 2:  # East
            next_position = (self.current_position[0], self.current_position[1] + 1)
        elif action == 3:  # West
            next_position = (self.current_position[0], self.current_position[1] - 1)

        if self._is_valid_move(next_position):
            self.current_position = next_position

        reward = self.grid[self.current_position[0], self.current_position[1]]
        done = self.current_position == self.goal_position
        return self.current_position[0] * self.grid_size + self.current_position[1], reward, done, {}

    def _is_valid_move(self, position):
        if position[0] < 0 or position[0] >= self.grid_size or position[1] < 0 or position[1] >= self.grid_size:
            return False
        if position in self.obstacle_positions:
            return False
        return True

    def render(self, mode='human'):
        print("Current Grid:")
        print(self.grid)


# Q-Learning, an off-policy temporal difference learning method

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# SARSA algorithm, an on-policy temporal difference learning method.

class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation

    def learn(self, state, action, reward, next_state, next_action, done):
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q * (1 - int(done))
        td_error = td_target - current_q
        new_value = current_q + self.alpha * td_error
        self.q_table[state, action] = new_value

env = GridWorldEnv()
q_learning_agent = QLearningAgent(env)
sarsa_agent = SarsaAgent(env)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    q_learning_done = False
    sarsa_done = False
    while not (q_learning_done and sarsa_done):
        # Q-Learning Agent
        q_learning_action = q_learning_agent.choose_action(state)
        next_state, reward, q_learning_done, _ = env.step(q_learning_action)
        q_learning_agent.learn(state, q_learning_action, reward, next_state, q_learning_done)
        state = next_state

        # SARSA Agent
        sarsa_action = sarsa_agent.choose_action(state)
        next_state, reward, sarsa_done, _ = env.step(sarsa_action)
        sarsa_agent.learn(state, sarsa_action, reward, next_state, sarsa_agent.choose_action(next_state), sarsa_done)
        state = next_state

print("Training Complete")

# Q-learning agent
q_learning_total_reward = 0
state = env.reset()
done = False
while not done:
    action = q_learning_agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    q_learning_total_reward += reward
    state = next_state
print("Q-Learning Total Reward:", q_learning_total_reward)

# SARSA agent
sarsa_total_reward = 0
state = env.reset()
done = False
while not done:
    action = sarsa_agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    sarsa_total_reward += reward
    state = next_state
print("SARSA Total Reward:", sarsa_total_reward)

#This setup allows for a direct comparison between Q-Learning and SARSA in this specific GridWorld environment.

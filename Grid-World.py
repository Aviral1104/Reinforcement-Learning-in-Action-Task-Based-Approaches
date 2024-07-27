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

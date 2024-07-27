'''
          GRID WORLD
   0  1  2  3  4  5  6  7  8  9
0 [S][ ][ ][ ][ ][ ][ ][ ][ ][ ]
1 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
2 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
3 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
4 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
5 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
6 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
7 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
8 [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
9 [ ][ ][ ][ ][ ][ ][ ][ ][ ][G]

  -The agent starts at the top-left corner (0,0).
  -The goal is at the bottom-right corner (9,9).
  -All states except the goal have a reward of -1.
  -The goal state has a reward of +100.
  -The agent can move up, down, left, or right from any position.
  -If an action would take the agent off the grid, it remains in its current position.
'''

import numpy as np

gamma = 0.8 
alpha = 0.5  
epsilon = 0.1 
grid_size = 10
state_space = grid_size * grid_size
action_space = 4  # up, down, left, right
Q = np.zeros((state_space, action_space))

R = -1 * np.ones((grid_size, grid_size))  # -1 reward for each step
R[9, 9] = 100  # +100 reward for reaching the goal


state_mapping = {(i, j): i * grid_size + j for i in range(grid_size) for j in range(grid_size)}
action_mapping = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right

# Q-learning algorithm
for episode in range(10000):
    state = (0, 0)
    
    while state != (9, 9):
        state_index = state_mapping[state]
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(action_space)  # explore
        else:
            action = np.argmax(Q[state_index])  # exploit
        new_state = (state[0] + action_mapping[action][0], state[1] + action_mapping[action][1])
        new_state = (max(0, min(grid_size - 1, new_state[0])), max(0, min(grid_size - 1, new_state[1])))
        reward = R[new_state]
        new_state_index = state_mapping[new_state]
        Q[state_index, action] = (1 - alpha) * Q[state_index, action] + alpha * (reward + gamma * np.max(Q[new_state_index]))
        state = new_state

print("Q-values:")
print(Q)

policy = np.argmax(Q, axis=1)
print("Policy (best actions from each state):")
print(policy)

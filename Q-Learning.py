'''
The environment has 8 states (0-7).
State transitions and rewards are defined by the matrices T and R.
The algorithm learns the optimal action to take in each state to maximize long-term rewards.
The final policy shows the best action to take from each state based on the learned Q-values.
'''

import numpy as np

# Transition matrix
T = np.array([
    [0, 2, 3, 0, 0, 0, 0, 0],
    [2, 0, 4, 5, 0, 0, 0, 0],
    [3, 4, 0, 7, 6 ,9, 0, 0],
    [0 ,5 ,7 ,0 ,8 ,10, 0, 0],
    [0 ,0 ,6 ,8 ,0 ,11, 0, 0],
    [0 ,0 ,0 ,0 ,0 ,0, 12, -9],
    [0 ,0 ,0 ,0 ,0 ,0, 0, -10],
    [0 ,0 ,0 ,0 ,0 ,0, 0, -11]
])

# Rewards matrix
R = np.copy(T)
R[R != -9] = -1

# Q-learning algorithm
Q = np.zeros_like(R)
gamma = .8 # discount factor
epsilon = 0.1 # exploration rate

for episode in range(1000):
    state = np.random.randint(8) 
    while state != -1:
        if np.random.uniform(0, 1) < epsilon:  # Epsilon-greedy policy
            action = np.random.choice(np.where(R[state] >= -1)[0])
        else:
            action = np.argmax(Q[state])
        
        future_rewards = []
        for next_possible_action in range(len(Q[action])):
            if R[action][next_possible_action] == -1:
                future_rewards.append(0)
            else:
                future_rewards.append(Q[action][next_possible_action])
        
        max_future_reward = max(future_rewards)
        
        Q[state][action] += R[state][action] + gamma * max_future_reward
        
        if R[state][action] == -9: 
            break
            
        state = action

print("Q-values:")
print(Q)

policy = np.argmax(Q, axis=1)
print("Policy (best actions from each state):")
print(policy)

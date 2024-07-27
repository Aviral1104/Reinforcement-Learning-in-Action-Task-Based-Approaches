import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random

#The QNetwork class for function approximation

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

#The DQNAgent class with methods for choosing actions, storing experiences, and learning from them
class DQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)
        self.update_target_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_network(np.expand_dims(state, axis=0))
            return np.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        target_q_values = self.target_q_network(next_states)
        max_target_q_values = np.max(target_q_values, axis=1)
        targets = rewards + self.gamma * max_target_q_values * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            predicted_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.reduce_mean(tf.square(targets - predicted_q_values))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# The main training loop that runs episodes, collects experiences, and periodically updates the target network
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
target_network_update_freq = 100
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple): 
        state = state[0]  
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Combine terminated and truncated flags
        total_reward += reward
        
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        state = next_state
    
    if episode % target_network_update_freq == 0:
        agent.update_target_network()
    
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        state = next_state
    # Update target Q-network weights
    if episode % target_network_update_freq == 0:
        agent.update_target_network()
    print("Episode:", episode, "Total Reward:", total_reward)

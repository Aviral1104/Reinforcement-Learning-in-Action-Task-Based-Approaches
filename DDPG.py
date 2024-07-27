'''
Implementation of Deep Deterministic Policy Gradient (DDPG) algorithm for reinforcement learning to solve the Pendulum-v1 environment in OpenAI Gym. 
DDPG is a model-free reinforcement learning algorithm that combines the strengths of actor-critic methods and deep neural networks.
two neural network models: Actor and Critic. The Actor model is responsible for mapping states to actions, while the Critic model evaluates the quality of the actions taken by the Actor. 
The code also includes a ReplayBuffer class to store and sample experiences (state, action, reward, next_state, done) during training.
'''

import random
import numpy as np
import tensorflow as tf
import gym
from collections import deque
from keras import layers

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_dim, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.concat_layer = layers.Concatenate()
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        state_input, action_input = inputs
        x = self.dense1(state_input)
        x = self.concat_layer([x, action_input])
        x = self.dense2(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        state, action, reward, next_state, done = experience
        self.buffer.append((np.array(state).flatten(), np.array(action).flatten(), reward, np.array(next_state).flatten(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(buffer_size=100000)
        
        # optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def choose_action(self, state):
        state = np.array(state).flatten().reshape(1, -1)  
        action = self.actor(state)
        noise = np.random.normal(0, 0.1, size=action.shape)
        return (action.numpy()[0] + noise) * 2  

    def update_target_networks(self, tau=0.001):
        for target_param, param in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)
        for target_param, param in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

    def update(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic([next_states, target_actions])
            target_q_values = rewards + (1 - dones) * gamma * target_q
            q_values = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            q_values = self.critic([states, actor_actions])
            actor_loss = -tf.reduce_mean(q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        self.update_target_networks()

        return actor_loss, critic_loss
      
'''
The main training loop iterates over 1000 episodes. In each episode, the agent interacts with the environment by choosing actions based on the current state and the Actor model.
The agent then observes the next state, reward, and whether the episode is done or truncated. The experience is stored in the ReplayBuffer, and once the buffer contains enough samples (batch_size), the agent updates the Actor and Critic models using the sampled experiences. 
The target networks (target_actor and target_critic) are also updated using a soft update mechanism to stabilize the learning process.
'''

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim)
batch_size = 64 

for episode in range(1000):
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        action = agent.choose_action(state)
        try:
            next_state, reward, done, truncated, _ = env.step(action)
        except ValueError as e:
            print(f"Error during environment step: {e}")
            break

        total_reward += reward

        agent.replay_buffer.add((state, action, reward, next_state, done))
        
        if len(agent.replay_buffer) > batch_size:
            experiences = agent.replay_buffer.sample(batch_size)
            actor_loss, critic_loss = agent.update(experiences)
            
        state = next_state
    
    print(f"Episode: {episode}, Total Reward: {total_reward.item():.2f}")

env.close()

'''
Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for reinforcement learning
Uses two critics to reduce overestimation bias
Adds noise to target actions for smoothing
Delays policy updates to reduce variance
It aims to learn an optimal policy for the Pendulum-v1 environment, where the goal is to swing up and balance an inverted pendulum.
'''
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.concat_layer = tf.keras.layers.Concatenate()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        state_input, action_input = inputs
        x = self.dense1(state_input)
        x = self.concat_layer([x, action_input])
        x = self.dense2(x)
        return self.output_layer(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)

        self.replay_buffer = ReplayBuffer(buffer_size=100000)
        self.soft_update(self.target_actor, self.actor, tau=1.0)  # Initialize target_actor
        self.soft_update(self.target_critic1, self.critic1, tau=1.0)  # Initialize target_critic1
        self.soft_update(self.target_critic2, self.critic2, tau=1.0)  # Initialize target_critic2

        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic1.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic2.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def choose_action(self, state):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)  
        action = self.actor(state_tensor)
        return np.clip(action.numpy()[0], -2, 2)  

    def update(self, experiences, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        states, actions, rewards, next_states, dones = experiences
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute target Q-values
        target_actions = self.target_actor(next_states)
        noise = tf.random.normal(shape=target_actions.shape, mean=0.0, stddev=policy_noise)
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        target_actions = tf.clip_by_value(target_actions + noise, -1, 1)
        
        target_q1 = self.target_critic1([next_states, target_actions])
        target_q2 = self.target_critic2([next_states, target_actions])
        target_q = tf.minimum(target_q1, target_q2)
        target_values = rewards + (1 - dones) * gamma * target_q

        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            q1_values = self.critic1([states, actions])
            q2_values = self.critic2([states, actions])
            critic1_loss = tf.reduce_mean(tf.square(q1_values - target_values))
            critic2_loss = tf.reduce_mean(tf.square(q2_values - target_values))
        critic1_gradients = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_gradients = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        del tape
        self.critic1.optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
        self.critic2.optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

        # Delayed policy updates
        if policy_freq == 0 or len(self.replay_buffer) % policy_freq == 0:
            with tf.GradientTape() as tape:
                actions_2d = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic1([states, actions_2d]))
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            # Soft update target networks
            self.soft_update(self.target_actor, self.actor, tau)
            self.soft_update(self.target_critic1, self.critic1, tau)
            self.soft_update(self.target_critic2, self.critic2, tau)

    def soft_update(self, target_model, source_model, tau):
        for target_param, source_param in zip(target_model.trainable_variables, source_model.trainable_variables):
            target_param.assign(tau * source_param + (1 - tau) * target_param)

# Main training loop
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3Agent(state_dim, action_dim)

for episode in range(1000):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0] 
    state = np.array(state, dtype=np.float32) 
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)  
        total_reward += reward
        
        agent.replay_buffer.add((state, action, reward, next_state, done))
        
        if len(agent.replay_buffer) > 64:  
            experiences = agent.replay_buffer.sample(64)
            agent.update(experiences)
        
        state = next_state
    
    print("Episode:", episode, "Total Reward:", total_reward)

env.close()

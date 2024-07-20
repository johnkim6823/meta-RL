import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from collections import deque
import gym

# Define SAC components
class SACAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.actor = self.create_actor()
        self.critic_1 = self.create_critic()
        self.critic_2 = self.create_critic()
        self.target_critic_1 = self.create_critic()
        self.target_critic_2 = self.create_critic()

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_1_optimizer = optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_optimizer = optimizers.Adam(learning_rate=critic_lr)

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 64

    def create_actor(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        return model

    def create_critic(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_dim + self.action_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[idx] for idx in batch])

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            target_actions = self.actor(next_states)
            target_q1 = self.target_critic_1(tf.concat([next_states, target_actions], axis=1))
            target_q2 = self.target_critic_2(tf.concat([next_states, target_actions], axis=1))
            target_q = tf.minimum(target_q1, target_q2)
            y = rewards + self.gamma * (1 - dones) * target_q
            q1 = self.critic_1(tf.concat([states, actions], axis=1))
            q2 = self.critic_2(tf.concat([states, actions], axis=1))
            critic_1_loss = tf.reduce_mean(tf.square(y - q1))
            critic_2_loss = tf.reduce_mean(tf.square(y - q2))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q1 = self.critic_1(tf.concat([states, actions], axis=1))
            actor_loss = -tf.reduce_mean(q1)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target(self.target_critic_1, self.critic_1)
        self.update_target(self.target_critic_2, self.critic_2)

    def update_target(self, target, source):
        for t, s in zip(target.trainable_variables, source.trainable_variables):
            t.assign(self.tau * s + (1 - self.tau) * t)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return self.actor(state)[0]

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

# Initialize environment
env = gym.make('CartPole-v1')  # Example environment

# Assuming state_dim and action_dim are defined
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # For discrete actions

sac_agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

# Load the meta-trained model weights
sac_agent.actor.load_weights('meta_trained_model_weights.h5')

# Interaction with the environment and SAC updates
num_episodes = 1000
max_steps = 200

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = sac_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        sac_agent.store(state, action, reward, next_state, done)
        sac_agent.update()
        state = next_state
        episode_reward += reward
        if done:
            break

    print(f'Episode: {episode}, Reward: {episode_reward}')

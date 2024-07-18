import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SAC:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

    def build_actor(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=-1)
        )
        return model

    def build_critic(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        return model

    def select_action(self, state):
        state = torch.FloatTensor(self.preprocess_state(state)).unsqueeze(0)
        action_probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=action_probs.detach().numpy()[0])
        return action

    def preprocess_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        return np.array(state, dtype=np.float32)

    def train(self, env, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                result = env.step(action)
                
                if len(result) >= 4:
                    next_state, reward, done, _ = result[:4]
                else:
                    raise ValueError(f"Unexpected number of elements returned by env.step: {len(result)}")

                episode_reward += reward
                state = next_state
            rewards.append(episode_reward)
        return rewards

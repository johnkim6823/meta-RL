# a3c.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class A3CPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(A3CPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std for Gaussian

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class A3CValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(A3CValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_a3c(env, policy_net, value_net, num_episodes=1000, gamma=0.99, lr=0.001, device='cpu'):
    policy_net.to(device)
    value_net.to(device)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            mean, std = policy_net(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample().clamp(-1, 1)
            scaled_action = env.action_space.low + 0.5 * (action + 1.0) * (env.action_space.high - env.action_space.low)
            log_prob = dist.log_prob(action).sum(dim=-1)

            next_state, reward, done, _ = env.step(scaled_action.cpu().numpy())
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            with torch.no_grad():
                next_value = value_net(next_state_tensor) if not done else 0
            value = value_net(state_tensor)
            advantage = reward + gamma * next_value - value

            policy_loss = -(log_prob * advantage.detach())
            value_loss = advantage.pow(2)

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards

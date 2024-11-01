#ddpg.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DDPGPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class DDPGQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_ddpg(env, policy_net, q_net, num_episodes=1000, gamma=0.99, lr=0.001, device='cpu'):
    policy_net.to(device)
    q_net.to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_q = optim.Adam(q_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            
            next_state, reward, done, _ = env.step(action)
            q_value = reward + gamma * q_net(torch.FloatTensor(next_state).unsqueeze(0).to(device), torch.FloatTensor(action).unsqueeze(0).to(device))
            loss_q = (q_value - q_net(state_tensor, torch.FloatTensor(action).unsqueeze(0).to(device))) ** 2

            optimizer_q.zero_grad()
            loss_q.backward()
            optimizer_q.step()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards

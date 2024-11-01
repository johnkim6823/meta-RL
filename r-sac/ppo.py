#ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class PPOValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_ppo(env, policy_net, value_net, num_episodes=1000, gamma=0.99, lr=0.001, device='cpu'):
    policy_net.to(device)
    value_net.to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            
            next_state, reward, done, _ = env.step(action)
            advantage = reward + gamma * value_net(torch.FloatTensor(next_state).unsqueeze(0).to(device)) - value_net(state_tensor)

            policy_loss = -torch.log(action) * advantage
            value_loss = advantage ** 2

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

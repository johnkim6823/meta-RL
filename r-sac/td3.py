#td3.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class TD3PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TD3PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class TD3QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TD3QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_td3(env, policy_net, q_net1, q_net2, num_episodes=1000, gamma=0.99, tau=0.005, lr=0.001, device='cpu'):
    policy_net.to(device)
    q_net1.to(device)
    q_net2.to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_q1 = optim.Adam(q_net1.parameters(), lr=lr)
    optimizer_q2 = optim.Adam(q_net2.parameters(), lr=lr)
    memory = []
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(memory) > 1000:
                batch = random.sample(memory, 64)
                
                states = torch.FloatTensor([x[0] for x in batch]).to(device)
                actions = torch.FloatTensor([x[1] for x in batch]).to(device)
                rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(device)
                next_states = torch.FloatTensor([x[3] for x in batch]).to(device)
                dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_actions = policy_net(next_states)
                    next_q1 = q_net1(next_states, next_actions)
                    next_q2 = q_net2(next_states, next_actions)
                    target_q = rewards + gamma * (1 - dones) * torch.min(next_q1, next_q2)

                q1 = q_net1(states, actions)
                q2 = q_net2(states, actions)
                loss_q1 = nn.MSELoss()(q1, target_q)
                loss_q2 = nn.MSELoss()(q2, target_q)

                optimizer_q1.zero_grad()
                loss_q1.backward()
                optimizer_q1.step()

                optimizer_q2.zero_grad()
                loss_q2.backward()
                optimizer_q2.step()
                
                predicted_actions = policy_net(states)
                q_pred = q_net1(states, predicted_actions)
                loss_policy = -q_pred.mean()

                optimizer_policy.zero_grad()
                loss_policy.backward()
                optimizer_policy.step()

                for target_param, param in zip(q_net1.parameters(), q_net1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(q_net2.parameters(), q_net2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards

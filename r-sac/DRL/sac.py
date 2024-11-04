import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, device='cpu'):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_sac(env, policy_net, q_net1, q_net2, num_episodes=1000, gamma=0.99, tau=0.005, lr=0.001, device='cpu'):
    # Move networks to the specified device
    policy_net.to(device)
    q_net1.to(device)
    q_net2.to(device)
    
    # Print device information for each network
    print("-" * 30)
    print(f"Policy Network is on device: {next(policy_net.parameters()).device}")
    print(f"Q Network 1 is on device: {next(q_net1.parameters()).device}")
    print(f"Q Network 2 is on device: {next(q_net2.parameters()).device}")
    print("-" * 30)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_q1 = optim.Adam(q_net1.parameters(), lr=lr)
    optimizer_q2 = optim.Adam(q_net2.parameters(), lr=lr)
    memory = []
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
        
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(memory) > 1000:
                batch = random.sample(memory, 64)
                
                states = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
                actions = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
                rewards = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
                dones = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1).to(device)

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

    print("SAC Training Completed")
    return episode_rewards

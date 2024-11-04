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


def train_ddpg(env, policy_net, q_net, num_episodes=1000, max_steps=200, gamma=0.99, lr=0.001, device='cpu'):
    policy_net.to(device)
    q_net.to(device)
    
    print("-" * 30)
    print(f"Policy Network is on device: {next(policy_net.parameters()).device}")
    print(f"Q Network is on device: {next(q_net.parameters()).device}")
    print("-" * 30)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_q = optim.Adam(q_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        
        episode_reward = 0
        done = False

        for step in range(max_steps):
            if done:
                break

            # Start timer
            state_tensor = torch.FloatTensor(state).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)

            next_state, reward, done, *rest = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            reward_tensor = torch.FloatTensor([reward]).to(device)

            # Calculate target Q-value and Q loss
            with torch.no_grad():
                target_q_value = reward_tensor + gamma * q_net(next_state_tensor, action_tensor)

            q_value = q_net(state_tensor, action_tensor)
            loss_q = (q_value - target_q_value).pow(2).mean()

            # Q-network update
            optimizer_q.zero_grad()
            loss_q.backward()
            optimizer_q.step()

            # Policy update
            policy_loss = -q_net(state_tensor, policy_net(state_tensor)).mean()
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Update state and reward
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards

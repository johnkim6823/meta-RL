import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class A2CPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(A2CPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Standard deviation parameter

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # Output mean within [-1, 1]
        std = self.log_std.exp().expand_as(mean)
        return mean, std  # Return both mean and std for the action distribution


class A2CValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(A2CValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def rescale_action(action, env):
    # Rescale action from [-1, 1] to the environment’s action space
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    scaled_action = action_low + (0.5 * (action + 1.0) * (action_high - action_low))
    return scaled_action

def train_a2c(env, policy_net, value_net, num_episodes=1000, gamma=0.99, lr=0.001, device='cpu'):
    policy_net.to(device)
    value_net.to(device)
    
    print("-" * 30)
    print(f"Policy Network is on device: {next(policy_net.parameters()).device}")
    print(f"Value Network is on device: {next(value_net.parameters()).device}")
    print("-" * 30)
        
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            mean, std = policy_net(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()  # Sample action from the normal distribution
            action = action.clamp(-1, 1)  # Clip action to stay within [-1, 1]

            # Rescale action to environment’s action space bounds
            scaled_action = rescale_action(action, env)
            log_prob = dist.log_prob(action).sum(dim=-1)

            next_state, reward, done, *info = env.step(scaled_action.cpu().numpy())
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            # Calculate advantage
            value = value_net(state_tensor)
            next_value = value_net(next_state_tensor)
            advantage = reward + gamma * next_value - value  # Keep gradient on value

            # Compute policy and value losses
            policy_loss = -(log_prob * advantage.detach())
            value_loss = advantage.pow(2)

            # Optimize policy network
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # Optimize value network
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards
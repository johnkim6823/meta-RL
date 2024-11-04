import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class PPOPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Log standard deviation for Gaussian

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # Ensuring mean is in [-1, 1] range
        std = self.log_std.exp().expand_as(mean)  # Standard deviation from log_std
        return mean, std  # Return mean and std for action distribution

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

def scale_action(action, action_space):
    """Rescale action from [-1, 1] to the action_space range."""
    low, high = action_space.low[0], action_space.high[0]
    return low + (0.5 * (action + 1.0) * (high - low))

def train_ppo(env, policy_net, value_net, num_episodes=1000, gamma=0.99, lr=0.0003, device='cpu'):
    # Move networks to the specified device
    policy_net.to(device)
    value_net.to(device)
    
    # Print device information for each network
    print(f"Policy Network is on device: {next(policy_net.parameters()).device}")
    print(f"Value Network is on device: {next(value_net.parameters()).device}")
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    episode_rewards = []

    for episode in range(num_episodes):
        # Reset environment and handle tuple output (state, info)
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract only the state if env.reset() returns (state, info)
        episode_reward = 0
        done = False

        while not done:
            # Ensure state is a numpy array and in the correct shape
            state = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state_tensor = torch.FloatTensor(state).to(device)

            # Obtain action distribution parameters (mean and std)
            mean, std = policy_net(state_tensor)
            dist = Normal(mean, std)
            raw_action = dist.sample()  # Sample action from distribution

            # Rescale action to environment's action space limits
            action = scale_action(torch.tanh(raw_action), env.action_space)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)  # Log prob of the unscaled action

            # Execute the action in the environment
            next_state, reward, done, *info = env.step(action.detach().cpu().numpy())
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Ensure next_state is also reshaped correctly
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            # Compute the advantage
            with torch.no_grad():
                next_value = value_net(next_state_tensor)
            advantage = reward + gamma * next_value - value_net(state_tensor)

            # Calculate policy and value loss
            policy_loss = -(log_prob * advantage.detach())  # Stop gradients on advantage
            value_loss = advantage.pow(2)

            # Backpropagate with gradient clipping for stability
            optimizer_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Clip gradients
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)  # Clip gradients
            optimizer_value.step()

            # Update state and episode reward
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    return episode_rewards

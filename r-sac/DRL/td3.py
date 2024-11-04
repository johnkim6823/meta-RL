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

    for episode in range(num_episodes):
        #print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle cases where env.reset() returns (state, info)
        state = np.asarray(state, dtype=np.float32)
        
        episode_reward = 0
        done = False

        while not done:
            # Convert state to tensor and sample action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).detach().cpu().numpy()[0]
            
            #print("Action sampled:", action)
            # Execute action in the environment
            next_state, reward, done, *info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Handle cases where env.step() returns (next_state, info)
            next_state = np.asarray(next_state, dtype=np.float32)

            # Store transition in memory
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # Train only if replay memory has enough samples
            if len(memory) > 1000:
                # Sample a batch from memory
                batch = random.sample(memory, 64)
                states = np.array([x[0] for x in batch], dtype=np.float32)
                actions = np.array([x[1] for x in batch], dtype=np.float32)
                rewards = np.array([x[2] for x in batch], dtype=np.float32).reshape(-1, 1)
                next_states = np.array([x[3] for x in batch], dtype=np.float32)
                dones = np.array([x[4] for x in batch], dtype=np.float32).reshape(-1, 1)

                # Convert numpy arrays to tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                with torch.no_grad():
                    next_actions = policy_net(next_states)
                    next_q1 = q_net1(next_states, next_actions)
                    next_q2 = q_net2(next_states, next_actions)
                    target_q = rewards + gamma * (1 - dones) * torch.min(next_q1, next_q2)
                #print("Target Q calculated.")

                # Compute Q loss
                q1 = q_net1(states, actions)
                q2 = q_net2(states, actions)
                loss_q1 = nn.MSELoss()(q1, target_q)
                loss_q2 = nn.MSELoss()(q2, target_q)
                #print("Q losses:", loss_q1.item(), loss_q2.item())

                # Backpropagate Q losses
                optimizer_q1.zero_grad()
                loss_q1.backward()
                optimizer_q1.step()

                optimizer_q2.zero_grad()
                loss_q2.backward()
                optimizer_q2.step()
                
                # Policy update
                predicted_actions = policy_net(states)
                q_pred = q_net1(states, predicted_actions)
                loss_policy = -q_pred.mean()
                #print("Policy loss:", loss_policy.item())

                optimizer_policy.zero_grad()
                loss_policy.backward()
                optimizer_policy.step()

                # Soft update target network
                for target_param, param in zip(q_net1.parameters(), q_net1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(q_net2.parameters(), q_net2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                #print("Target network updated.")

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} completed with Reward: {episode_reward}")

    print("TD3 Training Completed.")
    return episode_rewards

import torch
import torch.nn as nn
import torch.optim as optim
import random
from reptile_random import model  # Reptile 학습 모델 가져오기

# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

# Define the SAC class
class SAC:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = int(1e6)
        self.batch_size = 256

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        # Critic update
        with torch.no_grad():
            next_action = self.actor(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action
            target_q = reward + (1 - done) * self.gamma * next_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = (current_q1 - target_q).pow(2).mean() + (current_q2 - target_q).pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi = self.actor(state)
        q1_pi, q2_pi = self.critic(state, pi)
        actor_loss = (self.alpha * pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Soft update of target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.log_alpha, filename + "_log_alpha")
        torch.save(self.alpha_optim.state_dict(), filename + "_alpha_optim")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.log_alpha = torch.load(filename + "_log_alpha")
        self.alpha_optim.load_state_dict(torch.load(filename + "_alpha_optim"))

# Initialize SAC with the final Reptile model weights
state_dim = 1
action_dim = 1
max_action = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sac_agent = SAC(state_dim, action_dim, max_action)

# Copy weights from Reptile model to SAC actor
sac_agent.actor.l1.weight.data = model[0].weight.data.clone()
sac_agent.actor.l1.bias.data = model[0].bias.data.clone()

with torch.no_grad():
    sac_agent.actor.l2.weight.data[:64, :64] = model[2].weight.data.clone()
    sac_agent.actor.l2.bias.data = model[2].bias.data.clone()

print("SAC initialized with final Reptile weights.")

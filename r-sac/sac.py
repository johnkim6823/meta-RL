import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gym
import random

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if self.size < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).cuda(),
            torch.FloatTensor(action).cuda(),
            torch.FloatTensor(reward).cuda(),
            torch.FloatTensor(next_state).cuda(),
            torch.FloatTensor(1 - done).cuda()
        )

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic Network (Q1, Q2)
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

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = Critic(state_dim, action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.target_entropy = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp()

    def select_action(self, state):
        if isinstance(state, tuple):
            processed_state = []
            for s in state:
                if np.isscalar(s):
                    processed_state.append(np.array([s]))  # 0차원 스칼라를 1차원 배열로 변환
                elif isinstance(s, np.ndarray):
                    if s.ndim == 0:
                        processed_state.append(s.reshape(1))  # 0차원 배열을 1차원 배열로 변환
                    else:
                        processed_state.append(s)
                elif isinstance(s, dict):
                    # 딕셔너리의 모든 값을 추출하여 벡터로 변환
                    dict_values = [np.array(v).reshape(-1) for v in s.values() if len(np.array(v).shape) > 0]
                    if len(dict_values) > 0:
                        dict_values = np.concatenate(dict_values)
                        processed_state.append(dict_values)
                else:
                    # 상태의 요소가 배열, 스칼라, 딕셔너리가 아닌 경우 예외 발생
                    raise ValueError(f"State contains elements of unsupported type: {type(s)}")
            if len(processed_state) > 0:
                state = np.concatenate(processed_state, axis=-1)
            else:
                raise ValueError("Processed state is empty. Check the input state format.")
        else:
            state = np.array(state)
        
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action, _ = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Critic training
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor training
        pi, log_pi = self.actor(state)
        q1_pi, q2_pi = self.critic(state, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Test Agent
def test_agent(env, agent, episodes=10):
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)  # 학습된 정책을 사용하여 액션 선택
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) > 4:
                next_state, reward, done = result[0:3]
            else:
                raise ValueError("Unexpected number of values returned from env.step(action)")

            episode_reward += reward
            state = next_state

            env.render()  # 환경 렌더링 (시각화)

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")
    env.close()

# Main function
if __name__ == "__main__":
    # 환경 생성
    print("Initializing environment...")
    env = gym.make('Pendulum-v1')  # render_mode 제거
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # SAC 에이전트 생성
    print("Creating SAC Agent...")
    agent = SACAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_size=1000000)

    # 학습 및 테스트
    total_timesteps = 0
    max_timesteps = 1000000
    batch_size = 256
    start_timesteps = 10000
    eval_freq = 5000

    # 학습 시작
    print("Starting training...")
    while total_timesteps < max_timesteps:
        state = env.reset()
        episode_reward = 0
        done = False
        print(f"Starting episode at timestep {total_timesteps}...")

        while not done:
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()  # 초기에는 랜덤 액션
            else:
                action = agent.select_action(state)

            result = env.step(action)  # 모든 반환 값을 받음
            if len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) > 4:
                next_state, reward, done = result[0:3]
                info = result[3:]
            else:
                raise ValueError("Unexpected number of values returned from env.step(action)")

            done_bool = float(done) if episode_reward < 199 else 0

            # 리플레이 버퍼에 추가
            replay_buffer.add(state, action, reward, next_state, done_bool)

            state = next_state
            episode_reward += reward
            total_timesteps += 1

            # 학습 단계
            if total_timesteps >= start_timesteps:
                agent.train(replay_buffer, batch_size)
                print(f"Training at timestep {total_timesteps}")

            # 환경 렌더링
            env.render()

            # 평가 및 테스트
            if total_timesteps % eval_freq == 0:
                print(f"Evaluating at timestep {total_timesteps}...")
                test_agent(env, agent, episodes=10)

        print(f"Episode finished with reward {episode_reward}")

    # 모델 저장
    print("Saving model...")
    torch.save(agent.actor.state_dict(), 'actor.pth')
    torch.save(agent.critic.state_dict(), 'critic.pth')
    print("Model saved.")

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
from sklearn.metrics import mean_squared_error

# Import custom implementations of each algorithm
from sac import PolicyNetwork as SACPolicyNetwork, QNetwork as SACQNetwork, train_sac
from td3 import TD3PolicyNetwork, TD3QNetwork, train_td3
from ppo import PPOPolicyNetwork, PPOValueNetwork, train_ppo
from ddpg import DDPGPolicyNetwork, DDPGQNetwork, train_ddpg
from a2c import A2CPolicyNetwork, A2CValueNetwork, train_a2c

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# OpenAI Gym 환경 설정 (MountainCarContinuous 예제)
env = gym.make("MountainCarContinuous-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 결과 저장 폴더 생성
output_dir = "training_results_comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 이동 평균 함수 정의
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Hyperparameters
num_episodes = 500  # Reduced for quicker comparisons

# 1. SAC Training
print("🚀 Starting SAC Training...")
sac_policy_net = SACPolicyNetwork(state_dim, action_dim).to(device)
sac_q_net1 = SACQNetwork(state_dim, action_dim).to(device)
sac_q_net2 = SACQNetwork(state_dim, action_dim).to(device)
sac_rewards = train_sac(env, sac_policy_net, sac_q_net1, sac_q_net2, num_episodes=num_episodes, device=device)
print("✅ SAC Training Completed.")

# 2. TD3 Training
print("🚀 Starting TD3 Training...")
td3_policy_net = TD3PolicyNetwork(state_dim, action_dim).to(device)
td3_q_net1 = TD3QNetwork(state_dim, action_dim).to(device)
td3_q_net2 = TD3QNetwork(state_dim, action_dim).to(device)
td3_rewards = train_td3(env, td3_policy_net, td3_q_net1, td3_q_net2, num_episodes=num_episodes, device=device)
print("✅ TD3 Training Completed.")

# 3. PPO Training
print("🚀 Starting PPO Training...")
ppo_policy_net = PPOPolicyNetwork(state_dim, action_dim).to(device)
ppo_value_net = PPOValueNetwork(state_dim).to(device)
ppo_rewards = train_ppo(env, ppo_policy_net, ppo_value_net, num_episodes=num_episodes, device=device)
print("✅ PPO Training Completed.")

# 4. DDPG Training
print("🚀 Starting DDPG Training...")
ddpg_policy_net = DDPGPolicyNetwork(state_dim, action_dim).to(device)
ddpg_q_net = DDPGQNetwork(state_dim, action_dim).to(device)
ddpg_rewards = train_ddpg(env, ddpg_policy_net, ddpg_q_net, num_episodes=num_episodes, device=device)
print("✅ DDPG Training Completed.")

# 5. A2C Training
print("🚀 Starting A2C Training...")
a2c_policy_net = A2CPolicyNetwork(state_dim, action_dim).to(device)
a2c_value_net = A2CValueNetwork(state_dim).to(device)
a2c_rewards = train_a2c(env, a2c_policy_net, a2c_value_net, num_episodes=num_episodes, device=device)
print("✅ A2C Training Completed.")

# Compute MSE Loss for each algorithm compared to SAC
mse_td3 = mean_squared_error(sac_rewards[:num_episodes], td3_rewards[:num_episodes])
mse_ppo = mean_squared_error(sac_rewards[:num_episodes], ppo_rewards[:num_episodes])
mse_ddpg = mean_squared_error(sac_rewards[:num_episodes], ddpg_rewards[:num_episodes])
mse_a2c = mean_squared_error(sac_rewards[:num_episodes], a2c_rewards[:num_episodes])

print(f"\n=== MSE Loss compared to SAC ===")
print(f"TD3 MSE Loss: {mse_td3:.4f}")
print(f"PPO MSE Loss: {mse_ppo:.4f}")
print(f"DDPG MSE Loss: {mse_ddpg:.4f}")
print(f"A2C MSE Loss: {mse_a2c:.4f}")

# 학습 결과 시각화 (비교)
plt.figure(figsize=(12, 6))

# Plot SAC rewards
plt.plot(sac_rewards, label="SAC - Episode Rewards", alpha=0.3)
if len(sac_rewards) >= 100:
    avg_sac_rewards = moving_average(sac_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_sac_rewards, label="SAC - 100-Episode Moving Avg", linewidth=2)

# Plot TD3 rewards
plt.plot(td3_rewards, label="TD3 - Episode Rewards", alpha=0.3)
if len(td3_rewards) >= 100:
    avg_td3_rewards = moving_average(td3_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_td3_rewards, label="TD3 - 100-Episode Moving Avg", linewidth=2)

# Plot PPO rewards
plt.plot(ppo_rewards, label="PPO - Episode Rewards", alpha=0.3)
if len(ppo_rewards) >= 100:
    avg_ppo_rewards = moving_average(ppo_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_ppo_rewards, label="PPO - 100-Episode Moving Avg", linewidth=2)

# Plot DDPG rewards
plt.plot(ddpg_rewards, label="DDPG - Episode Rewards", alpha=0.3)
if len(ddpg_rewards) >= 100:
    avg_ddpg_rewards = moving_average(ddpg_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_ddpg_rewards, label="DDPG - 100-Episode Moving Avg", linewidth=2)

# Plot A2C rewards
plt.plot(a2c_rewards, label="A2C - Episode Rewards", alpha=0.3)
if len(a2c_rewards) >= 100:
    avg_a2c_rewards = moving_average(a2c_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_a2c_rewards, label="A2C - 100-Episode Moving Avg", linewidth=2)

# Finalize plot
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Algorithm Comparison on SAC, TD3, PPO, DDPG, A2C")
plt.legend()
plt.grid(True)

# Save the comparison plot
comparison_plot_path = os.path.join(output_dir, "algorithm_comparison.png")
plt.savefig(comparison_plot_path)
print(f"📊 Comparison plot saved at: {comparison_plot_path}")

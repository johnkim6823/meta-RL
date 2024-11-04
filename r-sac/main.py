import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import gym
from sklearn.metrics import mean_squared_error
import time  # To measure training duration

# Add the DRL folder to the system path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "DRL"))

# Import custom implementations of each algorithm from the DRL folder
from sac import PolicyNetwork as SACPolicyNetwork, QNetwork as SACQNetwork, train_sac
from td3 import TD3PolicyNetwork, TD3QNetwork, train_td3
from ppo import PPOPolicyNetwork, PPOValueNetwork, train_ppo
from ddpg import DDPGPolicyNetwork, DDPGQNetwork, train_ddpg
from a2c import A2CPolicyNetwork, A2CValueNetwork, train_a2c
from a3c import A3CPolicyNetwork, A3CValueNetwork, train_a3c
from reptile import reptile

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
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

# Create Reptile-initialized model
print("-" * 30)
print("🚀 Starting Reptile Meta-Learning for SAC Initialization...")

# Generate dummy tasks for Reptile (for simplicity)
def create_dummy_tasks(num_tasks=5, num_samples=10):
    tasks = []
    for _ in range(num_tasks):
        network_delay = torch.rand(num_samples, 1)
        cpu_usage = torch.rand(num_samples, 1)
        data = torch.cat([network_delay, cpu_usage], dim=1).to(device)
        labels = torch.rand(num_samples, action_dim).to(device)
        tasks.append((data, labels))
    return tasks

# Initialize a policy model for meta-learning
meta_policy_model = SACPolicyNetwork(state_dim, action_dim).to(device)
tasks = create_dummy_tasks(num_tasks=5, num_samples=10)

start_time = time.time()
reptile_initialized_model = reptile(meta_policy_model, tasks)
reptile_time = time.time() - start_time
print("✅ Reptile Meta-Learning Completed.")

# 1. Regular SAC Training (No Reptile)
print("-" * 30)
print("🚀 Starting Regular SAC Training...")
start_time = time.time()
sac_policy_net = SACPolicyNetwork(state_dim, action_dim).to(device)
sac_q_net1 = SACQNetwork(state_dim, action_dim).to(device)
sac_q_net2 = SACQNetwork(state_dim, action_dim).to(device)
sac_rewards = train_sac(env, sac_policy_net, sac_q_net1, sac_q_net2, num_episodes=num_episodes, device=device)
sac_time = time.time() - start_time
print("✅ Regular SAC Training Completed.")

# 2. SAC Training with Reptile-Initialized Model
print("-" * 30)
print("🚀 Starting SAC Training with Reptile-Initialized Model...")
start_time = time.time()
reptile_policy_net = reptile_initialized_model
sac_q_net1 = SACQNetwork(state_dim, action_dim).to(device)
sac_q_net2 = SACQNetwork(state_dim, action_dim).to(device)
reptile_sac_rewards = train_sac(env, reptile_policy_net, sac_q_net1, sac_q_net2, num_episodes=num_episodes, device=device)
reptile_sac_time = time.time() - start_time
print("✅ SAC Training with Reptile Initialization Completed.")

# 3. TD3 Training
print("-" * 30)
print("🚀 Starting TD3 Training...")
start_time = time.time()
td3_policy_net = TD3PolicyNetwork(state_dim, action_dim).to(device)
td3_q_net1 = TD3QNetwork(state_dim, action_dim).to(device)
td3_q_net2 = TD3QNetwork(state_dim, action_dim).to(device)
td3_rewards = train_td3(env, td3_policy_net, td3_q_net1, td3_q_net2, num_episodes=num_episodes, device=device)
td3_time = time.time() - start_time
print("✅ TD3 Training Completed.")

# 5. DDPG Training
print("-" * 30)
print("🚀 Starting DDPG Training...")
start_time = time.time()
ddpg_policy_net = DDPGPolicyNetwork(state_dim, action_dim).to(device)
ddpg_q_net = DDPGQNetwork(state_dim, action_dim).to(device)
ddpg_rewards = train_ddpg(env, ddpg_policy_net, ddpg_q_net, num_episodes=num_episodes, device=device)
ddpg_time = time.time() - start_time
print("✅ DDPG Training Completed.")

# 6. A2C Training
print("-" * 30)
print("🚀 Starting A2C Training...")
start_time = time.time()
a2c_policy_net = A2CPolicyNetwork(state_dim, action_dim).to(device)
a2c_value_net = A2CValueNetwork(state_dim).to(device)
a2c_rewards = train_a2c(env, a2c_policy_net, a2c_value_net, num_episodes=num_episodes, device=device)
a2c_time = time.time() - start_time
print("✅ A2C Training Completed.")

# 7. A3C Training
print("-" * 30)
print("🚀 Starting A3C Training...")
start_time = time.time()
a3c_policy_net = A3CPolicyNetwork(state_dim, action_dim).to(device)
a3c_value_net = A3CValueNetwork(state_dim).to(device)
a3c_rewards = train_a3c(env, a3c_policy_net, a3c_value_net, num_episodes=num_episodes, device=device)
a3c_time = time.time() - start_time
print("✅ A3C Training Completed.")

# Compute MSE Loss for each algorithm compared to Regular SAC
mse_reptile_sac = mean_squared_error(sac_rewards[:num_episodes], reptile_sac_rewards[:num_episodes])
mse_td3 = mean_squared_error(sac_rewards[:num_episodes], td3_rewards[:num_episodes])
mse_ddpg = mean_squared_error(sac_rewards[:num_episodes], ddpg_rewards[:num_episodes])
mse_a2c = mean_squared_error(sac_rewards[:num_episodes], a2c_rewards[:num_episodes])
mse_a3c = mean_squared_error(sac_rewards[:num_episodes], a3c_rewards[:num_episodes])

print(f"\n=== MSE Loss compared to Regular SAC ===")
print(f"Reptile-SAC MSE Loss: {mse_reptile_sac:.4f}")
print(f"TD3 MSE Loss: {mse_td3:.4f}")
print(f"DDPG MSE Loss: {mse_ddpg:.4f}")
print(f"A2C MSE Loss: {mse_a2c:.4f}")
print(f"A3C MSE Loss: {mse_a3c:.4f}")

# 학습 결과 시각화 (비교)
plt.figure(figsize=(12, 6))

# Plot Regular SAC rewards
plt.plot(sac_rewards, label="Regular SAC - Episode Rewards", alpha=0.3)
if len(sac_rewards) >= 100:
    avg_sac_rewards = moving_average(sac_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_sac_rewards, label="Regular SAC - 100-Episode Moving Avg", linewidth=2)

# Plot Reptile-SAC rewards
plt.plot(reptile_sac_rewards, label="Reptile-SAC - Episode Rewards", alpha=0.3)
if len(reptile_sac_rewards) >= 100:
    avg_reptile_sac_rewards = moving_average(reptile_sac_rewards, window_size=100)
    plt.plot(range(100, num_episodes + 1), avg_reptile_sac_rewards, label="Reptile-SAC - 100-Episode Moving Avg", linewidth=2)

# Plot other algorithms' rewards
plt.plot(td3_rewards, label="TD3 - Episode Rewards", alpha=0.3)
plt.plot(ddpg_rewards, label="DDPG - Episode Rewards", alpha=0.3)
plt.plot(a2c_rewards, label="A2C - Episode Rewards", alpha=0.3)
plt.plot(a3c_rewards, label="A3C - Episode Rewards", alpha=0.3)

# Finalize plot
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Algorithm Comparison")
plt.legend()
plt.grid(True)

# Save the comparison plot
comparison_plot_path = os.path.join(output_dir, "algorithm_comparison_with_reptile.png")
plt.savefig(comparison_plot_path)
print(f"📊 Comparison plot saved at: {comparison_plot_path}")

# Print training durations for each algorithm
print("\n=== Training Duration (seconds) ===")
print(f"Reptile Meta-Learning Time: {reptile_time:.2f} s")
print(f"Regular SAC Training Time: {sac_time:.2f} s")
print(f"SAC Training with Reptile Initialization Time: {reptile_sac_time:.2f} s")
print(f"TD3 Training Time: {td3_time:.2f} s")
print(f"DDPG Training Time: {ddpg_time:.2f} s")
print(f"A2C Training Time: {a2c_time:.2f} s")
print(f"A3C Training Time: {a3c_time:.2f} s")

print("Training comparison completed.")

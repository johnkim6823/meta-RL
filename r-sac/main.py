import numpy as np
import matplotlib.pyplot as plt
from sac import SAC
from reptile import Reptile
from tasks import get_tasks  # tasks.py에서 get_tasks 함수 가져오기

def plot_rewards(rewards, title="Training Rewards"):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def main():
    sac = SAC(state_dim=4, action_dim=2)  # CartPole-v1의 state_dim과 action_dim 사용
    reptile = Reptile(sac)

    num_tasks = 10
    tasks = get_tasks(num_tasks)  # tasks.py에서 작업 로드
    
    all_rewards, new_meta_policy = reptile.train(tasks)
    
    # 훈련 에피소드 동안의 평균 보상 출력
    for i, rewards in enumerate(all_rewards):
        average_reward = np.mean(rewards)
        print(f'Average Reward during Training for Task {i+1}: {average_reward}')
    
    # 훈련 보상 플로팅
    for i, rewards in enumerate(all_rewards):
        plot_rewards(rewards, title=f"Training Rewards for Task {i+1}")
    
    # 평가 에피소드 실행
    sac.actor.load_state_dict(new_meta_policy)  # Load the new meta policy
    num_eval_episodes = 10
    eval_rewards = []
    for _ in range(num_eval_episodes):
        env, _ = tasks[0]  # 첫 번째 환경 사용
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = sac.select_action(state)
            result = env.step(action)
            
            if len(result) >= 4:
                state, reward, done, _ = result[:4]
            else:
                raise ValueError(f"Unexpected number of elements returned by env.step: {len(result)}")

            episode_reward += reward
        eval_rewards.append(episode_reward)
    
    # 평가 에피소드 동안의 평균 보상 출력
    eval_average_reward = np.mean(eval_rewards)
    print(f'Average Reward during Evaluation: {eval_average_reward}')

if __name__ == '__main__':
    main()

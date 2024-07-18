import torch

class Reptile:
    def __init__(self, sac, num_inner_updates=5, step_size=0.01):
        self.sac = sac
        self.num_inner_updates = num_inner_updates
        self.step_size = step_size

    def inner_update(self, sac, env, num_episodes):
        rewards = sac.train(env, num_episodes)
        return sac.actor.state_dict(), rewards

    def outer_update(self, meta_policy, task_policies):
        # Aggregate updates to the meta-policy
        new_meta_policy = meta_policy.copy()
        for key in meta_policy.keys():
            updates = [task_policy[key] for task_policy in task_policies]
            new_meta_policy[key] = torch.mean(torch.stack(updates), dim=0)
        return new_meta_policy

    def train(self, tasks):
        meta_policy = self.sac.actor.state_dict()
        all_rewards = []
        task_policies = []
        for task in tasks:
            env, num_episodes = task
            self.sac.actor.load_state_dict(meta_policy)
            task_policy, rewards = self.inner_update(self.sac, env, num_episodes)
            task_policies.append(task_policy)
            all_rewards.append(rewards)
        new_meta_policy = self.outer_update(meta_policy, task_policies)
        self.sac.actor.load_state_dict(new_meta_policy)
        return all_rewards, new_meta_policy

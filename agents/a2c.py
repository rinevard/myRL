import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from infrastructure import utils

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, ac_dim):
        super().__init__()
        hidden_dim = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_dim, ac_dim)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"Input must be a tensor!"
        assert len(x.shape) > 1, f"Input must be batched!"
        return F.softmax(self.net(x), dim=-1)

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        hidden_dim = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor), f"Input must be a tensor!"
        assert len(x.shape) > 1, f"Input must be batched!"
        return self.net(x)
    
class A2CAgent:
    def __init__(self, env: gym.Env, discounted_factor: float=0.95, 
                 actor_lr: float=1e-4, critic_lr: float=1e-4):
        """
        连续观测, 离散动作
        """
        self.env = env
        self.discounted_factor = discounted_factor

        action_dim = self.env.action_space.n
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            obs_dim = np.int64(1)
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            obs_dim = self.env.observation_space.shape[0]
        self.actor = PolicyNet(obs_dim, action_dim)
        self.critic = ValueNet(obs_dim)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def get_action(self, obs, is_training=True):
        assert isinstance(obs, np.ndarray), f"Input must be an ndarray, but get obs: {obs}"
        obs = utils.from_numpy(obs)
        obs = torch.unsqueeze(obs, dim=0)
        action_dists =  self.actor(obs).squeeze(0)
        if is_training:
            action = torch.multinomial(action_dists, num_samples=1)[0]
        else:
            action = torch.argmax(action_dists)
        action = utils.to_numpy(action)
        
        assert action in self.env.action_space, f"Returned action {action} not in action space!"
        return action

    def update(self, observations, actions, rewards, terminates, next_observations):
        """
        Input must be batched numpy arrays.
        """
        observations = utils.from_numpy(observations)
        actions = utils.from_numpy(actions)
        rewards = utils.from_numpy(rewards)
        terminates = utils.from_numpy(terminates)
        next_observations = utils.from_numpy(next_observations)
        actions = actions.to(dtype=torch.int64)

        # 归一化观测值
        observations = (observations - observations.mean()) / (observations.std() + 1e-8)
        next_observations = (next_observations - next_observations.mean()) / (next_observations.std() + 1e-8)
        
        # 归一化奖励
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # update critic
        with torch.no_grad():
            td_targets =  rewards + self.discounted_factor * (1 - terminates) * self.critic(next_observations)
            td_errors = td_targets - self.critic(observations)

        critic_loss = F.mse_loss(self.critic(observations), td_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actions_dists = self.actor(observations)
        actions_probs = actions_dists.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        actions_log_probs = torch.log(actions_probs + 1e-10)
        advantages = td_errors.detach()
        actor_loss = (-advantages * actions_log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()
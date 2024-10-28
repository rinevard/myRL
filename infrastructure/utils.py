from collections import OrderedDict
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Tuple, List

############################################
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_trajectory(
    env: gym.Env, agent, max_length: int) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, info = env.reset()
    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    steps = 0
    while True:
        action = agent.get_action(ob)

        next_ob, reward, terminated, truncated, info = env.step(action)

        steps += 1
        done: bool = (terminated or truncated or steps >= max_length)

        obs.append(ob)
        acs.append(action)
        rewards.append(reward)
        next_obs.append(next_ob)
        terminals.append(done)

        ob = next_ob 

        if done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }

def sample_trajectories(
    env: gym.Env,
    agent,
    min_timesteps_per_batch: int,
    max_length: int
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, agent, max_length)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch

def sample_n_trajectories(
    env: gym.Env, agent, ntraj: int, max_length: int):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, agent, max_length)
        trajs.append(traj)
    return trajs

def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )

def get_traj_length(traj):
    return len(traj["reward"])

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor: torch.Tensor):
    return tensor.to('cpu').detach().numpy()
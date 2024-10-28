import numpy as np
import gymnasium as gym
from collections import defaultdict

class DiscreteQlearningAgent:
    def __init__(self, env: gym.Env, learning_rate: float=0.01,
                 epsilon_start: float=1.0, epsilon_decay: float=0.005, epsilon_end: float=0.05,
                   discounted_factor: float=0.99):
        self.env = env
        self.lr = learning_rate

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

        self.discounted_factor = discounted_factor
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs) -> int:
        """
        Return random action with probability 'epsilon' and return argmax_a Q(obs, a) otherwise.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        td_target = reward + (not terminated) * self.discounted_factor * np.max(self.q_values[next_obs])
        td_error = td_target - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
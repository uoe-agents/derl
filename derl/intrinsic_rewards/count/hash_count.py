import numpy as np
import torch

from derl.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from derl.utils.utils import flatten


class HashCount(IntrinsicReward):
    """
    Hash-based count bonus for exploration class

    Paper:
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, O. X., Duan, Y., ... & Abbeel, P. (2017).
    # Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in neural information processing systems (pp. 2753-2762).

    Paper: https://arxiv.org/abs/1611.04717

    Open-source code: https://github.com/openai/EPG/blob/master/epg/exploration.py
    """
    def __init__(
        self,
        observation_space,
        action_space,
        parallel_envs,
        cfg,
        **kwargs,
    ):
        """
        Initialise parameters for hash count intrinsic reward
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs: number of parallel environments
        :param config: intrinsic reward configuration dict
        """
        super(HashCount, self).__init__(observation_space, action_space, parallel_envs, cfg)
        # Hashing function: SimHash
        if self.bucket_sizes is None:
            # Large prime numbers
            self.bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in self.bucket_sizes:
            mod = 1
            mods = []
            for _ in range(self.key_dim):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(self.bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))
        self.projection_matrix = np.random.normal(size=(self.obs_size, self.key_dim))

    def __compute_keys(self, obss):
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        keys = np.cast["int"](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def __inc_hash(self, obss):
        keys = self.__compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], self.decay_factor)

    def __query_hash(self, obss):
        keys = self.__compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.__query_hash(obss)
        self.__inc_hash(obss)

    def __predict(self, obs):
        counts = self.__query_hash(obs)
        __prediction = 1.0 / np.maximum(1.0, np.sqrt(counts))
        return __prediction

    def compute_intrinsic_reward(self, state, action, next_state, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        state = flatten(state)
        state = state.detach().cpu().numpy()
        if train:
            self.fit_before_process_samples(state)
        reward = torch.from_numpy(self.__predict(state)).float().to(self.device)

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * reward,
        }

    def reset(self):
        """
        Reset counting
        """
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))

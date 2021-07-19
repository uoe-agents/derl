from collections import defaultdict
from math import sqrt

import torch

from derl.intrinsic_rewards.intrinsic_reward import IntrinsicReward


class DictCount(IntrinsicReward):
    """
    Simple dict-based counting of observations

    Open-source (based on train_state_count in): https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/utils.py#L192
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
        Initialise parameters for dict-based count intrinsic reward
        :param observation_space: space of observation input
        :param action_space: space of action input
        :param parallel_envs: number of parallel environments
        :param config: intrinsic reward configuration dict
        """
        super(DictCount, self).__init__(observation_space, action_space, parallel_envs, cfg)
        self.count_table = defaultdict(int)

    def compute_intrinsic_reward(self, state, action, next_state, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        state_flat = [tuple(s.view(-1).tolist()) for s in state]
        rewards = []
        for s in state_flat:
            if train:
                self.count_table[s] += self.decay_factor
            count = self.count_table[s]
            reward = 1.0 / (self.decay_factor * max(1.0, sqrt(count)))
            rewards.append(reward)
        rewards = torch.FloatTensor(rewards).to(self.device)

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * rewards,
        }

    def reset(self):
        """
        Reset counting
        """
        self.count_table = defaultdict(int)

import copy

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from derl.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from derl.intrinsic_rewards.ride.model import RIDENetwork

from derl.intrinsic_rewards.count.dict_count import DictCount
from derl.intrinsic_rewards.count.hash_count import HashCount


class RIDE(IntrinsicReward):
    """
    Rewarding Impact-Driven Exploration (RIDE) code

    Paper:
    Raileanu, Roberta, and Tim Rocktäschel (2020).
    RIDE: Rewarding impact-driven exploration for procedurally-generated environments.
    In International Conference on Learning Representations.

    Paper: https://arxiv.org/abs/2002.12292
	
	Open-source code: https://github.com/facebookresearch/impact-driven-exploration
    """
    def __init__(self, observation_space, action_space, parallel_envs, cfg, **kwargs):
        """
        Initialise parameters for RIDE intrinsic reward definition
        :param observation_space (gym.spaces.space): observation space of environment
        :param action space (gym.spaces.space): action space of environment
        :param parallel_envs (int): number of parallel environments
        :param cfg (Dict): configuration for intrinsic reward
        """
        super(RIDE, self).__init__(observation_space, action_space, parallel_envs, cfg)
        self.discrete_actions = isinstance(action_space, gym.spaces.Discrete)

        # create model architecture
        self.model = RIDENetwork(
            self.observation_space, self.action_size, cfg.model
        ).to(self.model_device)
        
        # define episodic count (reset at each episode so separate counts for each parallel environment)
        # don't use intrinsic reward coef for episodic counts
        count_config = dict(copy.deepcopy(cfg.model.count))
        count_config["intrinsic_reward_coef"] = 1.0
        count_config["device"] = self.model_device
        if self.model_count_type == "hash":
            self.episodic_counts = [HashCount(observation_space, action_space, parallel_envs, count_config) for _ in range(parallel_envs)]
        elif self.model_count_type == "dict":
            self.episodic_counts = [DictCount(observation_space, action_space, parallel_envs, count_config) for _ in range(parallel_envs)]
        else:
            raise ValueError(f"Unknown episodic count {self.model_count_type} for RIDE.")

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prediction(self, state, action, next_state):
        """
        Compute prediction
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :return: (batch of) control_rewards, count_rewards, forward loss, inverse_loss
        """
        actions_onehot = torch.eye(self.action_size)[action.long()].squeeze().to(self.model_device)
        state_rep, next_state_rep, predicted_action, predicted_next_state_rep = self.model(
            state, next_state, actions_onehot
        )
        if self.discrete_actions and predicted_action.shape == actions_onehot.shape:
            # discrete one-hot encoded action
            action_targets = actions_onehot.max(1)[1]
            inverse_loss = F.cross_entropy(predicted_action, action_targets, reduction="none")
        else:
            inverse_loss = ((predicted_action - actions_onehot) ** 2).sum(-1)
        forward_loss = 0.5 * ((next_state_rep - predicted_next_state_rep) ** 2).sum(-1)

        control_rewards = 0.5 * ((next_state_rep - state_rep) ** 2).sum(-1)
        count_rewards = torch.FloatTensor([
            episodic_count.compute_intrinsic_reward(
                state[i].unsqueeze(0),
                action[i].unsqueeze(0),
                next_state[i].unsqueeze(0),
            )["intrinsic_reward"]
            for i, episodic_count in enumerate(self.episodic_counts)
        ]).to(self.model_device)

        # number of parallel environments
        assert state.shape[0] == action.shape[0] == next_state.shape[0] == self.parallel_envs == control_rewards.shape[0] == count_rewards.shape[0]

        return control_rewards, count_rewards, forward_loss, inverse_loss

    def compute_intrinsic_reward(self, state, action, next_state, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param train: flag if model should be trained
        :return: dict of 'intrinsic reward' and losses
        """
        if train:
            control_rewards, count_rewards, forward_loss, inverse_loss = self._prediction(state, action, next_state)
        else:
            with torch.no_grad():
                control_rewards, count_rewards, forward_loss, inverse_loss = self._prediction(state, action, next_state)

        loss = self.forward_loss_coef * forward_loss.mean() + self.inverse_loss_coef * inverse_loss.mean()

        # optimise RIDE model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        intrinsic_rewards = count_rewards * control_rewards
        int_reward = intrinsic_rewards.detach()

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * int_reward,
            "forward_loss": forward_loss.mean(),
            "inverse_loss": inverse_loss.mean(),
            "control_rewards": control_rewards.mean(),
            "count_rewards": count_rewards.mean(),
        }

    def episode_reset(self, environment_idx):
        """
        Indicate termination of episode/ start of new episode

        :param environment_idx: index of environment for which new episode started
        """
        self.episodic_counts[environment_idx].reset()

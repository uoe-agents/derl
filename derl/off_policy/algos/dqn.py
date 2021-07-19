import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from derl.off_policy.algorithm import Algorithm
from derl.off_policy.common.model import QNetwork, DuelingQNetwork
from derl.off_policy.common.replay_buffer import NStepReplayBuffer
from derl.off_policy.common.prioritised_replay_buffer import NStepPrioritisedReplayBuffer
from derl.off_policy.common.utils import soft_update, hard_update, epsilon_scheduler


class DQN(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        parallel_envs,
        num_env_steps,
        cfg,
        **kwargs,
    ):
        super(DQN, self).__init__(observation_space, action_space, cfg)
        self.parallel_envs = parallel_envs
        self.num_acts = action_space.n
        if self.model_architecture == "default":
            self.model = QNetwork(
                observation_space, action_space, cfg.model
            )
            self.target_model = QNetwork(
                observation_space, action_space, cfg.model
            )
        elif self.model_architecture == "dueling":
            self.model = DuelingQNetwork(
                observation_space, action_space, cfg.model
            )
            self.target_model = DuelingQNetwork(
                observation_space, action_space, cfg.model
            )
        else:
            raise ValueError(f"Invalid DQN architecture {self.model_architecture}!")

        hard_update(self.target_model, self.model)
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model.to(self.model_device)
        self.target_model.to(self.model_device)

        if self.replay_buffer_type == "default":
            self.memory = NStepReplayBuffer(self.buffer_capacity, self.n_steps, parallel_envs, self.gamma, self.use_proper_time_limits)
        elif self.replay_buffer_type == "prioritised":
            self.memory = NStepPrioritisedReplayBuffer(observation_space.shape, action_space.n, num_env_steps, self.buffer_capacity, self.n_steps, parallel_envs, self.gamma, self.use_proper_time_limits)
        else:
            raise ValueError(f"Invalid replay buffer type {self.replay_buffer_type}!")

        self.optimiser = optim.Adam(self.model.parameters(), self.lr, eps=self.adam_eps)

        self.saveables = {
            "model": self.model,
            "target_model": self.target_model,
            "optimiser": self.optimiser,
        }

        self.epsilon_scheduler = epsilon_scheduler(self.eps_start, self.eps_end, self.eps_decay)
        self.steps = 0
        self.epsilon = self.epsilon_scheduler(0)

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    def init_training(self, obs):
        """
        Initalise training storage

        :param envs (VecEnv): vectorised environment
        :param obss (torch.Tensor): tensor of initial observation of shape (num_envs, obs_dim)
        """
        pass

    def insert_experience(
        self,
        obs,
        mask,
        n_obs,
        actions,
        action_log_prob,
        value,
        rewards,
        masks,
        bad_masks,
    ):
        targets = self._compute_targets(n_obs)
        for parallel_env, (o, act, rew, tar, d, b_d) in enumerate(zip(obs, actions, rewards, targets, masks, bad_masks)):
            self.memory.add(parallel_env, o, act, rew, tar, d, b_d)

    def _compute_targets(self, n_obs):
        """
        :param n_obs (torch.Tensor): tensor observations (batch_size, obs_size)
        :return (torch.Tensor): target values for next observation
        """
        with torch.no_grad():
            if self.targets == "default":
                q_next_states = self.target_model(n_obs)
                target_next_states = q_next_states.max(-1)[0]
            elif self.targets == "double":
                q_tp1_values = self.model(n_obs).detach()
                _, a_prime = q_tp1_values.max(1)
                q_next_states = self.target_model(n_obs)
                target_next_states = q_next_states.gather(1, a_prime.unsqueeze(1))
        return target_next_states

    def compute_returns(self):
        pass

    def act(self, obs, mask, evaluation=False):
        """
        Choose action for agent given observation

        :param obs: observation to act in
        :param mask: action mask
        :param evaluation: boolean whether action selection is for evaluation
        :return: state values, actions, action log-probs
        """
        with torch.no_grad():
            actions = self.epsilon_greedy(obs, evaluation)
        return None, actions, None

    def epsilon_greedy(self, obs, evaluation=False):
        """
        Epsilon-greedy action selection
        :param obs (torch.Tensor): observation vector
        :param evaluation (bool): boolean flag whether action selection is for evaluation
        :return (torch.Tensor): chosen action
        """
        self.steps += 1
        self.epsilon = self.epsilon_scheduler(self.steps)

        if evaluation:
            epsilon = self.greedy_epsilon
        else:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            # random action
            action = torch.Tensor(np.random.choice(range(self.num_acts), size=obs.shape[0])).int()
        else:
            # greedy action
            qvals = self.model(obs)
            action = qvals.argmax(dim=-1).int()
        return action

    def update(self, beh_update, other_policy=None):
        """
        Compute and execute update

        :param beh_update: boolean whether update for behaviour policy (True) or exploitation
            policy (False)
        :param other_policy: model of behaviour policy (if exploitation policy is trained) or
            model of exploitation policy (if exploration policy is trained)
        :return: dictionary of losses
        """
        if len(self.memory) < self.batch_size:
            # not enough samples yet to update
            return {}

        if self.replay_buffer_type == "default":
            obs, act, ret = self.memory.sample(self.batch_size)
        elif self.replay_buffer_type == "prioritised":
            obs, act, ret, ind, weights = self.memory.sample(self.batch_size)
        target_states = ret.detach()

        # local Q-values
        all_q_states = self.model(obs)
        q_states = all_q_states.gather(1, act.long())

        if self.replay_buffer_type == "default":
            loss = F.mse_loss(q_states, target_states)
        elif self.replay_buffer_type == "prioritised":
            loss = (weights * F.mse_loss(q_states, target_states, reduction='none')).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        soft_update(self.target_model, self.model, self.tau)

        if self.replay_buffer_type == "prioritised":
            priority = ((q_states - target_states).abs() + self.replay_buffer_prioritised_increment_epsilon)
            priority = priority.pow(self.replay_buffer_prioritised_exponent_alpha).cpu().data.numpy().flatten()
            self.memory.update_priority(ind, priority)

        return {
            "q_loss": loss.item()
        }

    def after_update(self):
        """
        Post update processing
        """
        pass

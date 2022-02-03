import os

import torch
import torch.nn as nn
import torch.optim as optim

from derl.on_policy.algorithm import Algorithm
from derl.on_policy.common.model import Policy
from derl.on_policy.common.storage import RolloutStorage
from derl.utils.utils import kl_divergence


class A2C(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        parallel_envs,
        num_env_steps,
        cfg,
        **kwargs,
    ):
        super(A2C, self).__init__(observation_space, action_space, cfg)

        self.model = Policy(
            observation_space, action_space, cfg.model.actor, cfg.model.critic, cfg.model.activation,
        )

        self.storage = RolloutStorage(
            observation_space,
            action_space,
            self.n_steps,
            parallel_envs,
        )

        self.model.to(self.model_device)
        self.optimiser = optim.Adam(self.model.parameters(), self.lr, eps=self.adam_eps)

        self.saveables = {
            "model": self.model,
            "optimiser": self.optimiser,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    def act(self, obs, mask, evaluation=False):
        """
        Choose action for agent given observation (always uses stochastic policy greedy)

        :param obs: observation to act in
        :param mask: action mask
        :param evaluation: boolean whether action selection is for evaluation
        :return: state values, actions, action log-probs (for all envs)
        """
        with torch.no_grad():
            value, action, action_log_prob = self.model.act(
                obs,
                mask,
                deterministic=evaluation if self.greedy_evaluation else False,
            )
        return value, action, action_log_prob

    def insert_experience(
        self,
        obs,
        mask,
        n_obs,
        action,
        action_log_prob,
        value,
        reward,
        masks,
        bad_masks,
    ):
        if value is None:
            # value for exploitation policy has to be computed first
            value = self.model.get_value(obs, mask).detach()
        self.storage.insert(
            n_obs,
            action,
            action_log_prob,
            value,
            reward[:],
            masks,
            bad_masks,
        )

    def compute_returns(self):
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.masks[-1],
            ).detach()

        self.storage.compute_returns(
            next_value, self.use_gae, self.gamma, self.gae_lambda, self.use_proper_time_limits,
        )

    def update(self, beh_update, other_policy=None):
        """
        Compute and execute update

        :param beh_update: boolean whether update for behaviour policy (True) or exploitation
            policy (False)
        :param other_policy: other algorithm - behaviour policy (if exploitation policy is trained) or
            exploitation policy (if exploration policy is trained)
        :return: dictionary of losses
        """
        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        n_steps, parallel_envs, _ = self.storage.rewards.size()

        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )

        values = values.view(n_steps, parallel_envs, 1)
        action_log_probs = action_log_probs.view(n_steps, parallel_envs, 1)

        advantages = self.storage.returns[:-1] - values

        # for exploitation policy compute importance sampling weights
        if not beh_update:
            _, behavioural_action_log_probs, _ = other_policy.model.evaluate_actions(
                self.storage.obs[:-1].view(-1, *obs_shape),
                self.storage.masks[:-1].view(-1, 1),
                self.storage.actions.view(-1, action_shape),
            )
            behavioural_action_log_probs = behavioural_action_log_probs.view(n_steps, parallel_envs, 1)

            # compute importance sampling weights for training of exploitation model
            importance_sampling = (
                action_log_probs.exp() / (behavioural_action_log_probs.exp() + 1e-7)
            ).detach()

            if self.importance_sampling == "default":
                importance_sampling = importance_sampling
            elif self.importance_sampling == "retrace":
                ones = torch.ones_like(importance_sampling).to(self.model_device)
                importance_sampling = torch.min(ones, importance_sampling)
            else:
                raise ValueError(f"Invalid importance sampling configuration '{self.importance_sampling}'")

            value_loss = (importance_sampling * advantages.pow(2)).mean()
            policy_loss = -(importance_sampling * advantages.detach() * action_log_probs).mean()
        else:
            value_loss = advantages.pow(2).mean()
            policy_loss = -(advantages.detach() * action_log_probs).mean()
            importance_sampling = None

        if self.kl_coef != 0.0 and other_policy is not None:
            # compute KL divergence of policies | KL(pi_e || pi_beta)
            other_log_policy = other_policy.evaluate_policy_distribution(
                self.storage.obs[:-1].view(-1, *obs_shape),
                self.storage.masks[:-1].view(-1, 1),
            )
            other_log_policy = other_log_policy.detach()
            log_policy = self.evaluate_policy_distribution(
                self.storage.obs[:-1].view(-1, *obs_shape),
                self.storage.masks[:-1].view(-1, 1),
            )
            kl = kl_divergence(log_policy, other_log_policy).mean()
        else:
            kl = 0.0

        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * dist_entropy
            + self.kl_coef * kl
        )

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        loss_dict = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "dist_entropy": dist_entropy.item(),
        }

        if importance_sampling is not None:
            loss_dict["importance_sampling_weights"] = importance_sampling.mean().item()

        if kl != 0.0:
            loss_dict["kl_divergence"] = kl.item()

        return loss_dict

    def after_update(self):
        """
        Post update processing
        """
        self.storage.after_update()

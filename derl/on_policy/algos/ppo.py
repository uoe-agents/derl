import os

import torch
import torch.nn as nn
import torch.optim as optim

from derl.on_policy.algorithm import Algorithm
from derl.on_policy.common.model import Policy
from derl.on_policy.common.storage import RolloutStorage
from derl.utils.utils import kl_divergence


class PPO(Algorithm):
    def __init__(
        self,
        observation_space,
        action_space,
        parallel_envs,
        num_env_steps,
        cfg,
        **kwargs,
    ):
        super(PPO, self).__init__(observation_space, action_space, cfg)
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
        advantages = self.storage.returns[:-1] - self.storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        dist_kl_epoch = 0

        for e in range(self.num_epochs):
            data_generator = self.storage.feed_forward_generator(
                advantages, self.num_minibatches)

            for sample in data_generator:
                obs_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.model.evaluate_actions(
                    obs_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                if self.kl_coef != 0.0 and other_policy is not None:
                    # compute KL divergence of policies | KL(pi_e || pi_beta)
                    other_log_policy = other_policy.evaluate_policy_distribution(
                        obs_batch, masks_batch,
                    )
                    other_log_policy = other_log_policy.detach()
                    log_policy = self.evaluate_policy_distribution(
                        obs_batch, masks_batch
                    )
                    kl = kl_divergence(log_policy, other_log_policy).mean()
                else:
                    kl = torch.tensor(0.0).to(self.model_device)

                self.optimiser.zero_grad()
                (
                    action_loss
                    + value_loss * self.value_loss_coef
                    - dist_entropy * self.entropy_coef
                    + kl * self.kl_coef
                ).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimiser.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                dist_kl_epoch += kl.item()
        

        num_updates = self.num_epochs * self.num_minibatches

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        dist_kl_epoch /= num_updates

        loss_dict = {
            "policy_loss": action_loss_epoch,
            "value_loss": value_loss_epoch,
            "dist_entropy": dist_entropy_epoch,
        }

        if dist_kl_epoch != 0.0:
            loss_dict["kl_divergence"] = dist_kl_epoch

        return loss_dict

    def after_update(self):
        """
        Post update processing
        """
        self.storage.after_update()

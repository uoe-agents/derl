import torch
import torch.optim as optim

from derl.intrinsic_rewards.intrinsic_reward import IntrinsicReward
from derl.intrinsic_rewards.rnd.model import RNDNetwork


class RND(IntrinsicReward):
    """
    Random Network Distillation (RND) class

    Paper:
    Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
    Exploration by random network distillation.
    arXiv preprint arXiv:1810.12894.

    Paper: https://arxiv.org/abs/1810.12894

	Open-source code: https://github.com/openai/random-network-distillation
    """
    def __init__(self, observation_space, action_space, parallel_envs, cfg, **kwargs):
        """
        Initialise parameters for RND intrinsic reward definition
        :param observation_space: observation space of environment
        :param action space: action space of environment
        :param parallel_envs: number of parallel environments
        :param cfg: configuration for intrinsic reward
        """
        super(RND, self).__init__(observation_space, action_space, parallel_envs, cfg)

        # create models
        self.predictor_model = RNDNetwork(observation_space, cfg.model).to(self.model_device)
        self.target_model = RNDNetwork(observation_space, cfg.model).to(self.model_device)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.predictor_model.parameters(), lr=self.lr)

    def compute_intrinsic_reward(self, state, action, next_state, train=True):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: use CUDA tensors
        :return: dict of 'intrinsic reward' and losses
        """
        if train:
            target_feature = self.target_model(next_state)
            predict_feature = self.predictor_model(next_state)

            forward_loss = ((target_feature - predict_feature) ** 2).sum(-1)
        else:
            with torch.no_grad():
                target_feature = self.target_model(next_state)
                predict_feature = self.predictor_model(next_state)
                forward_loss = ((target_feature - predict_feature) ** 2).sum(-1)
        

        loss = self.rnd_loss_coef * forward_loss.mean()

        # optimise RND model
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), self.max_grad_norm)
        self.optimiser.step()

        int_reward = forward_loss.detach()

        return {
            "intrinsic_reward": self.intrinsic_reward_coef * int_reward,
            "rnd_loss": forward_loss.mean(),
        }

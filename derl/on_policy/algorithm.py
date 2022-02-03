from abc import ABC, abstractmethod

from gym.spaces.utils import flatdim

from derl.utils.utils import flatten_dict as flatten


class Algorithm(ABC):
    def __init__(
        self,
        observation_space,
        action_space,
        algorithm_config,
    ):
        self.obs_space = observation_space
        self.action_space = action_space
        self.config = algorithm_config

        self.obs_size = flatdim(observation_space)
        self.action_size = flatdim(action_space)

        # set all values from config as attributes
        for k, v in flatten(algorithm_config).items():
            setattr(self, k, v)

    @abstractmethod
    def save(self, path):
        raise NotImplementedError
        
    @abstractmethod
    def restore(self, path):
        raise NotImplementedError

    def init_training(self, obs):
        """
        Initalise training storage

        :param envs (VecEnv): vectorised environment
        :param obss (torch.Tensor): tensor of initial observation of shape (num_envs, obs_dim)
        """
        self.storage.obs[0].copy_(obs)
        self.storage.to(self.model_device)

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def compute_returns(self):
        raise NotImplementedError
        
    @abstractmethod
    def act(self, obs, mask, evaluation=False):
        """
        Choose action for agent given observation

        :param obs: observation to act in
        :param mask: action mask
        :param evaluation: boolean whether action selection is for evaluation
        :return: extrinsic state values, intrinsic state values, actions, action log-probs (for all envs)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """
        Compute and execute update

        :return: dictionary of losses
        """
        raise NotImplementedError

    def evaluate_policy_distribution(self, obs, masks):
        """
        Compute log-probs of policy across all actions
        :param obs: batch of observation to evaluate policy in
        :param masks: batch of mask for policy input
        :return: pytorch tensor of shape (N, |A|) with N = batchsize and |A| = number of actions
            including log-probs for each state of batch and each action
        """
        return self.model.evaluate_policy_distribution(obs, masks)

    @abstractmethod
    def after_update(self):
        """
        Post update processing
        """
        raise NotImplementedError

import torch
import torch.nn as nn

from derl.on_policy.common.model import prod
from derl.utils.utils import build_sequential


class RNDNetwork(nn.Module):
    """Random Network Distillation (RND) network"""

    def __init__(self, observation_space, model_dict):
        """
        Initialize parameters and build model.
        :param observation_space: space of each observation
        :param model_dict: dictionary for model configuration
        """
        super(RNDNetwork, self).__init__()
        self.observation_space = observation_space
        self.state_rep_size = model_dict.state_representation[-1]
        input_dim = prod(observation_space.shape)

        # state representation
        self.state_rep = build_sequential(input_dim, model_dict.state_representation)

    def forward(self, state):
        """
        Compute forward pass over RND network
        :param state: state
        :return: state representation
        """
        return self.state_rep(state)

import copy

import torch.nn as nn
import torch.nn.functional as F

from derl.utils.utils import build_sequential, prod


class QNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(
        self, obs_space, action_space, model_config,
    ):
        """
        Initialize parameters and build model.
        :param obs_space: observation space of environment
        :param action_space: action space of environment
        :param model_config: configuration of model
        """
        super(QNetwork, self).__init__()
        obs_shape = obs_space.shape
        state_size = prod(obs_shape)
        action_size = action_space.n

        hiddens = copy.deepcopy(model_config.hiddens)
        hiddens.append(action_size)
        self.network = build_sequential(state_size, hiddens, model_config.activation, output_activation=False)

    def forward(self, state):
        """
        Compute forward pass over QNetwork
        :param state: state representation for input state
        :return: forward pass result
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """Deep Q-Network with dueling architecture"""

    def __init__(
        self, obs_space, action_space, model_config,
    ):
        """
        Initialize parameters and build model.
        :param obs_space: observation space of environment
        :param action_space: action space of environment
        :param model_config: configuration of model
        """
        super(DuelingQNetwork, self).__init__()
        obs_shape = obs_space.shape
        state_size = prod(obs_shape)
        action_size = action_space.n

        # set common feature layer
        self.feature_layer = build_sequential(state_size, model_config.hiddens, output_activation=True)
        
        # set advantage layer
        advantage_hiddens = copy.deepcopy(model_config.advantage)
        advantage_hiddens.append(action_size)
        self.advantage_layer = build_sequential(model_config.hiddens[-1], advantage_hiddens, output_activation=False)

        # set value layer
        value_hiddens = copy.deepcopy(model_config.value)
        value_hiddens.append(1)
        self.value_layer = build_sequential(model_config.hiddens[-1], value_hiddens, output_activation=False)

    def forward(self, state):
        """
        Compute forward pass over QNetwork
        :param state: state representation for input state
        :return: forward pass result
        """
        feature = self.feature_layer(state)
        
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q

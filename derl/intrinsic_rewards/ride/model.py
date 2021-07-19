import torch
import torch.nn as nn

from derl.on_policy.common.model import prod
from derl.utils.utils import build_sequential


class RIDENetwork(nn.Module):
    """Network for Rewarding Impact-Driven Exploration (RIDE)"""

    def __init__(
        self, observation_space, action_size, model_dict,
    ):
        """
        Initialize parameters and build model.
        :param observation_space: space of each observation
        :param action_size: dimension of each action
        :param model_dict: dictionary for model configuration
        """
        super(RIDENetwork, self).__init__()
        self.state_rep_size = model_dict.state_representation[-1]

        input_dim = prod(observation_space.shape)

        # state representation
        self.state_rep = build_sequential(input_dim, model_dict.state_representation)

        # inverse model
        inverse_model_hiddens = model_dict.inverse_model
        inverse_model_hiddens.append(action_size)
        self.inverse_model = build_sequential(self.state_rep_size * 2, inverse_model_hiddens)

        # forward model
        forward_model_hiddens = model_dict.forward_model
        forward_model_hiddens.append(self.state_rep_size)
        self.forward_model = build_sequential(self.state_rep_size + action_size, forward_model_hiddens)

    def forward(self, state, next_state, action):
        """
        Compute forward pass over RIDE network
        :param state: current state
        :param next_state: reached state
        :param action: applied action
        :return: 
            state representation for current state,
            state representation for next state, 
            predicted_action,
            predicted state representation for next state, 
        """
        # compute state representations
        state_rep = self.state_rep(state)
        next_state_rep = self.state_rep(next_state)

        # inverse model output
        inverse_input = torch.cat([state_rep, next_state_rep], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # forward model output
        forward_input = torch.cat([state_rep, action], dim=1)
        predicted_next_state_rep = self.forward_model(forward_input)

        return state_rep, next_state_rep, predicted_action, predicted_next_state_rep

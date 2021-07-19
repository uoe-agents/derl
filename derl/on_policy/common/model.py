import torch
import torch.nn as nn

from derl.on_policy.common.distributions import Categorical
from derl.utils.utils import init_, build_sequential, Flatten, prod


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, actor_hiddens, critic_hiddens, activation, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        obs_shape = obs_space.shape
        input_dim = prod(obs_shape)

        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(input_dim, actor_hiddens, critic_hiddens, activation)

        self.num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, self.num_outputs)

    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
    
    def evaluate_policy_distribution(self, inputs, masks):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        policy_log_probs = []
        for a in range(self.num_outputs):
            actions = torch.ones(inputs.shape[0],) * a
            action_log_probs = dist.log_probs(actions).detach()
            policy_log_probs.append(action_log_probs)
        policy_log_probs = torch.stack(policy_log_probs, dim=1).squeeze()
        return value, policy_log_probs

class NNBase(nn.Module):
    def __init__(self, input_size, actor_hiddens, critic_hiddens):
        super(NNBase, self).__init__()

        self._input_size = input_size
        self._actor_hiddens = actor_hiddens
        self._critic_hiddens = critic_hiddens

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._actor_hiddens[-1]


class MLPBase(NNBase):
    def __init__(self, num_inputs, actor_hiddens, critic_hiddens, activation):
        super(MLPBase, self).__init__(num_inputs, actor_hiddens, critic_hiddens)

        self.actor = build_sequential(num_inputs, actor_hiddens, activation)
        self.critic = build_sequential(num_inputs, critic_hiddens, activation)

        self.critic_linear = init_(nn.Linear(critic_hiddens[-1], 1))

        self.train()

    def forward(self, inputs, masks):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor

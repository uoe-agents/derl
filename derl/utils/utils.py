import collections

import numpy as np
import torch.nn as nn


def flatten(tensor):
    return tensor.view(tensor.size(0), -1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def prod(iterable):
    p = 1
    for i in iterable:
        p *= i
    return p


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)


def build_sequential(num_inputs, hiddens, activation="relu", output_activation=True):
    modules = [Flatten()]
    if activation == "relu":
        nonlin = nn.ReLU
    elif activation == "tanh":
        nonlin = nn.Tanh
    else:
        raise ValueError(f"Unknown activation option {activation}!")
    
    assert len(hiddens) > 0
    modules.append(init_(nn.Linear(num_inputs, hiddens[0])))
    for i in range(len(hiddens) - 1):
        modules.append(nonlin())
        modules.append(init_(nn.Linear(hiddens[i], hiddens[i + 1])))
    if output_activation:
        modules.append(nonlin())
    return nn.Sequential(*modules)


def kl_divergence(p_logs, q_logs):
	"""
	Compute KL divergence between log policies
	:param p_logs: pytorch tensor of shape (N, |A|) with N = batchsize and |A| = nmber of actions
		including log-probs for each state of batch and each action 
	:param q_logs: pytorch tensor of shape (N, |A|) with N = batchsize and |A| = nmber of actions
		including log-probs for each state of batch and each action 
	:return: pytorch tensor of shape (N,) with KL divergence KL(p || q) for each batch 
	"""
	kl = (p_logs.exp() * (p_logs - q_logs)).sum(-1)
	return kl


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    keys.discard("episode")
    # save episode statistics
    for k in ["r", "l", "t"]:
        values = [d["episode"][k] for d in info]
        mean = np.mean(values)
        new_info[k] = mean
    for key in keys:
        mean = np.mean([d[key] for d in info if key in d])
        new_info[key] = mean
    return new_info

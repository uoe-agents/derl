# MODIFIED FROM: https://github.com/sfujim/LAP-PAL/blob/master/discrete/utils.py#L119

from collections import deque

import numpy as np
import torch


# Replay buffer for standard gym tasks
class PrioritisedReplayBuffer():
    def __init__(self, state_shape, action_dim, training_steps, buffer_size):
        self.max_size = int(buffer_size)

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, *state_shape))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.done_mask = np.zeros((self.max_size, 1))

        self.tree = SumTree(self.max_size)
        self.max_priority = 1.0
        self.beta = 0.4
        self.beta_increment_per_sampling = (1.0 - self.beta) / training_steps

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_mask[self.ptr] = done

        self.tree.set(self.ptr, self.max_priority)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, idxes=None):
        """
        Sample a batch of experiences.
        :param batch_size (int): How many transitions to sample
        :param idxes (List[int]): list of indeces or None
        :returns: 
        obs_batch (torch.Tensor): batch of observations
        act_batch (torch.Tensor): batch of actions executed given obs_batch
        ret_batches (torch.Tensor): batch of returns
        """
        if idxes is None:
            ind = self.tree.sample(batch_size)
        else:
            ind = idxes

        batch = (
            torch.FloatTensor(self.state[ind]),
            torch.LongTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done_mask[ind]),
        )

        weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
        weights /= weights.max()
        self.beta = min(self.beta + self.beta_increment_per_sampling, 1)
        batch += (ind, torch.FloatTensor(weights).reshape(-1, 1))
        return batch

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

    def __len__(self):
        return len(self.size)


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)
        
        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]
            
            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater
        
        return node_index

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]
        
        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


class NStepPrioritisedReplayBuffer(object):
    def __init__(self, state_shape, action_dim, training_steps, buffer_size, n_step, num_processes, gamma, use_proper_time_limits=True):
        """
        Create n-step replay buffer.
        :param state_shape (Tuple[int]): shape of states
        :param action_dim (int): dimensionality of actions
        :param training_steps (int): number of steps executed during entire training
        :param buffer_size (int): Max number of transitions to store in the buffer. When the buffer
        overflows the old memories are dropped.
        :param n_step (int): n-steps
        :param num_processes (int): number of parallel processes
        :param gamma (float): discount factor gamma
        :param use_proper_time_limits (bool): flag whether artificially time-ended episode done
        should be treated as episode end (False) in n-step return computation or only original
        dones are respected (cut off episodes are estimated for the "rest" using target
        estimates)
        """
        # initialise PER
        self.max_size = int(buffer_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, *state_shape))
        self.action = np.zeros((self.max_size, 1))
        self.action.shape
        self.next_state = np.array(self.state)
        self.returns = np.zeros((self.max_size, 1))

        self.tree = SumTree(self.max_size)
        self.max_priority = 1.0
        self.beta = 0.4
        self.beta_increment_per_sampling = (1.0 - self.beta) / training_steps

        # initialise n-step elements
        self.n_step_storages = [deque(maxlen=n_step + 1) for _ in range(num_processes)]
        self.n_step = n_step
        self.num_processes = num_processes
        self.gamma = gamma
        self.use_proper_time_limits = use_proper_time_limits

    def __len__(self):
        return self.size
    
    def _get_nstep_data(self, idx):
        """
        Compute n-step return for 1st entry of ith nstep_storage
        :param i (int): index of n-step storage
        :return:
        """
        # G set to last step target
        target = self.n_step_storages[idx][-1][3]
        G = target

        for i in reversed(range(1, self.n_step + 1)):
            _, _, rew, target, d_mask, bad_d_mask = self.n_step_storages[idx][i]
            if self.use_proper_time_limits:
                prev_target = self.n_step_storages[idx][i - 1][3]
                G = (
                (rew + self.gamma * d_mask * G) * bad_d_mask
                + prev_target * (1 - bad_d_mask)
                )
            else:
                G = rew + self.gamma * d_mask * G

        # take obs, action of first step index 0 just for last from last update
        # used for prev_target in case of time limits
        obs, action = self.n_step_storages[idx][1][:2]
        return obs, action, G

    def add(self, process_index, obs, action, reward, target, done_mask, bad_done_mask):
        data = (obs.clone(), action.clone(), reward.clone(), target.clone(), done_mask.clone(), bad_done_mask.clone())

        # compute n-step returns if already enough steps
        self.n_step_storages[process_index].append(data)
        if len(self.n_step_storages[process_index]) < self.n_step + 1:
            # not yet enough samples
            return
        obs, action, G = self._get_nstep_data(process_index)

        # add data to PER
        self.state[self.ptr] = obs
        self.action[self.ptr] = action
        self.returns[self.ptr] = G

        self.tree.set(self.ptr, self.max_priority)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, idxes=None):
        """
        Sample a batch of experiences.
        :param batch_size (int): How many transitions to sample
        :param idxes (List[int]): list of indeces or None
        :returns: 
        obs_batch (torch.Tensor): batch of observations
        act_batch (torch.Tensor): batch of actions executed given obs_batch
        ret_batches (torch.Tensor): batch of returns
        """
        if idxes is None:
            ind = self.tree.sample(batch_size)
        else:
            ind = idxes

        batch = (
            torch.FloatTensor(self.state[ind]).view(batch_size, -1),
            torch.LongTensor(self.action[ind]),
            torch.FloatTensor(self.returns[ind]),
        )

        weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
        weights /= weights.max()
        self.beta = min(self.beta + self.beta_increment_per_sampling, 1)
        batch += (ind, torch.FloatTensor(weights).reshape(-1, 1))
        return batch

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)
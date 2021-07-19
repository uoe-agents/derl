from collections import deque
import random

import torch


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, bad_done):
        data = (obs_t, action, reward, obs_tp1, done, bad_done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, bad_dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, bad_done = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
            bad_dones.append(bad_done)
        return torch.vstack(obses_t), torch.vstack(actions), torch.vstack(rewards), torch.vstack(obses_tp1), torch.vstack(dones), torch.vstack(bad_dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class NStepReplayBuffer(object):
    def __init__(self, size, n_step, num_processes, gamma, use_proper_time_limits=True):
        """
        Create n-step replay buffer.
        :param size (int): Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        :param n_step (int): n-steps
        :param num_processes (int): number of parallel processes
        :param gamma (float): discount factor gamma
        :param use_proper_time_limits (bool): flag whether artificially time-ended episode done
            should be treated as episode end (False) in n-step return computation or only original
            dones are respected (cut off episodes are estimated for the "rest" using target
            estimates)
        """
        self._storage = []
        self.n_step_storages = [deque(maxlen=n_step + 1) for _ in range(num_processes)]
        self._maxsize = int(size)
        self._next_idx = 0
        self.n_step = n_step
        self.num_processes = num_processes
        self.gamma = gamma
        self.use_proper_time_limits = use_proper_time_limits

    def __len__(self):
        return len(self._storage)
    
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
        self.n_step_storages[process_index].append(data)

        if len(self.n_step_storages[process_index]) < self.n_step + 1:
            # not yet enough samples
            return
        data_nstep = self._get_nstep_data(process_index)

        if self._next_idx >= len(self._storage):
            self._storage.append(data_nstep)
        else:
            self._storage[self._next_idx] = data_nstep
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, actions, returns = [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, ret = data
            obses.append(obs.view(-1))
            actions.append(action)
            returns.append(ret)
        return torch.vstack(obses), torch.vstack(actions), torch.vstack(returns)

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
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

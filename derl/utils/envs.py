import os

import gym
import numpy as np
import torch
from gym import ObservationWrapper
from gym.spaces.box import Box
from gym.wrappers import Monitor, RecordEpisodeStatistics

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import bsuite
from bsuite.utils import gym_wrapper
import hallway_explore


def make_bsuite_env(
    env_id=bsuite.sweep.DEEP_SEA,
    logging_dir=None,
):
    if logging_dir is not None:
        assert os.path.isdir(logging_dir)
        env = bsuite.load_and_record_to_csv(env_id, results_dir=logging_dir)
    else:
        env = bsuite.load_from_id(env_id)
    gym_env = gym_wrapper.GymFromDMEnv(env)
    return gym_env


def make_env(env_id, seed, rank, wrappers, monitor_dir):
    def _thunk():
        if env_id.startswith("bsuite"):
            bsuite_id = env_id.split("-")[1]
            env = make_bsuite_env(bsuite_id)
        else:
            env = gym.make(env_id)
        if seed is not None:
            env.seed(seed + rank)

        for wrapper in wrappers:
            if isinstance(wrapper, str):
                env = getattr(gym.wrappers, wrapper)(env)
            else:
                env = wrapper(env)
        
        if monitor_dir:
            env = Monitor(env, monitor_dir, lambda ep: int(ep==0), force=True, uid=str(rank))

        return env

    return _thunk


def make_vec_envs(
    name, dummy_vecenv, norm_obs, norm_rewards, gamma, seed, parallel_envs, wrappers, device
):
    envs = [
        make_env(name, seed, i, wrappers, None) for i in range(parallel_envs)
    ]

    if dummy_vecenv or len(envs) == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs, start_method="fork")

    if len(envs.observation_space.shape) == 1:
        if norm_rewards:
            envs = VecNormalize(envs, norm_obs=norm_obs, gamma=gamma)
        else:
            envs = VecNormalize(envs, norm_obs=norm_obs, norm_reward=False)

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        return torch.from_numpy(obs).to(self.device)

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return (
            torch.from_numpy(obs).float().to(self.device),
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )

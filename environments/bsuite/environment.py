import os
import random

import bsuite
from bsuite.utils import gym_wrapper

def make_bsuite_env(
    env_id="deep_sea/0",
    logging_dir=None,
):
    if logging_dir is not None:
        assert os.path.isdir(logging_dir)
        env = bsuite.load_and_record_to_csv(env_id, results_dir=logging_dir)
    else:
        env = bsuite.load_from_id(env_id)
    gym_env = gym_wrapper.GymFromDMEnv(env)
    return gym_env

if __name__ == "__main__":
    for i in range(21):
        env_id = f"deep_sea/{i}"
        env = make_bsuite_env(env_id)

        num_actions = env.action_space.n
        print(env_id, env.observation_space, num_actions)
        done = False
        obs = env.reset()
        t = 0
        while not done:
            action = random.randint(0, num_actions - 1)
            n_obs, r, done, _ = env.step(action)
            t += 1

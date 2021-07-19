import random

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg)

    torch.set_num_threads(1)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    env = hydra.utils.call(cfg.env, gamma=cfg.algorithm.gamma, seed=cfg.seed)

    exploration_agent = hydra.utils.instantiate(
        cfg.algorithm,
        observation_space=env.observation_space,
        action_space=env.action_space,
        parallel_envs=cfg.env.parallel_envs,
        num_env_steps=cfg.algorithm.num_env_steps,
        cfg=cfg.algorithm,
    )

    if "exploitation_algorithm" in cfg and cfg.exploitation_algorithm is not None:
        exploitation_agent = hydra.utils.instantiate(
            cfg.exploitation_algorithm,
            observation_space=env.observation_space,
            action_space=env.action_space,
            parallel_envs=cfg.env.parallel_envs,
            num_env_steps=cfg.algorithm.num_env_steps,
            cfg=cfg.exploitation_algorithm,
        )
    else:
        exploitation_agent = None

    if "curiosity" in cfg:
        curiosity = hydra.utils.instantiate(
            cfg.curiosity,
            observation_space=env.observation_space,
            action_space=env.action_space,
            parallel_envs=cfg.env.parallel_envs,
            cfg=cfg.curiosity,
        )
    else:
        curiosity = None
    
    hydra.utils.call(cfg, envs=env, exploration_agent=exploration_agent, exploitation_agent=exploitation_agent, curiosity_instance=curiosity, logger_instance=logger, cfg=cfg)

if __name__ == "__main__":
    main()

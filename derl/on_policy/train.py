import os
import time
from collections import deque

import torch

from derl.utils.envs import make_vec_envs
from derl.utils.utils import _squash_info


def evaluate(
	envs,
    agent,
    episodes_per_eval,
    device,
):
    envs.training = False

    n_obs = envs.reset().float()
    n_masks = torch.zeros(episodes_per_eval, 1).float().to(device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, action, _ = agent.act(n_obs, n_masks, evaluation=True)

        # Obser reward and next obs
        n_obs, _, done, infos = envs.step(action)

        n_masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done]
        ).to(device)
        for info, d in zip(infos, done):
            if d:
                all_infos.append(info)

    envs.training = True
    return all_infos

def main(
    envs,
    exploration_agent,
    exploitation_agent,
    curiosity_instance,
    logger_instance,
    cfg,
    **kwargs,
):

    curiosity = curiosity_instance
    logger = logger_instance
    obs = envs.reset()
    exploration_agent.init_training(obs)

    if exploitation_agent is not None:
        exploitation_agent.init_training(obs)

    start = time.time()
    num_updates = (
        int(cfg.algorithm.num_env_steps) // cfg.algorithm.n_steps // cfg.env.parallel_envs
    )

    all_infos = deque(maxlen=10)

    total_steps = 0
    completed_episodes = 0

    last_eval_ep = 0

    for n_updates in range(1, num_updates + 1):
        for step in range(cfg.algorithm.n_steps):
            # Sample actions
            obs = exploration_agent.storage.obs[step]
            mask = exploration_agent.storage.masks[step]
            value, action, action_log_prob = exploration_agent.act(obs, mask, evaluation=False)
            # Obser reward and next obs
            next_obs, reward, done, infos = envs.step(action)

            if curiosity is not None:
                # compute intrinsic reward
                int_dict = curiosity.compute_intrinsic_reward(obs, action, next_obs)

                int_reward = int_dict["intrinsic_reward"]
                int_dict["intrinsic_reward"] = int_reward.mean()
                logger.log_intrinsic_reward(int_dict, total_steps)
            else:
                int_reward = torch.zeros_like(reward)

            logger.log_extrinsic_reward(reward.mean(), total_steps)

            if cfg.train_extrinsic_intrinsic_rewards:
                reward += int_reward

            if cfg.train_intrinsic_extrinsic_rewards:
                int_reward += reward


            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )

            exploration_agent.insert_experience(
                obs,
                mask,
                next_obs,
                action,
                action_log_prob,
                value,
                int_reward[:].unsqueeze(1) if exploitation_agent is not None else reward[:].unsqueeze(1),
                masks,
                bad_masks,
            )

            if exploitation_agent is not None:
                exploitation_agent.insert_experience(
                    obs,
                    mask,
                    next_obs,
                    action,
                    action_log_prob,
                    None,
                    reward[:].unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for i, info in enumerate(infos):
                if info:
                    completed_episodes += 1
                    all_infos.append(info)
                    logger.log_episode(completed_episodes, info)
                    if curiosity is not None:
                        curiosity.episode_reset(i)

            total_steps += cfg.env.parallel_envs

        exploration_agent.compute_returns()
        if exploitation_agent is not None:
            exploitation_agent.compute_returns()

        loss_dict = exploration_agent.update()

        if exploitation_agent is not None:
            if n_updates % cfg.exploitation_algorithm.update_freq == 0:
                exp_loss_dict = exploitation_agent.update(exploration_agent.model)
                for k, v in exp_loss_dict.items():
                    loss_dict[f"exploitation_{k}"] = v

        logger.log_update(total_steps, n_updates)
        logger.log_losses(loss_dict, total_steps)

        exploration_agent.after_update()
        if exploitation_agent is not None:
            exploitation_agent.after_update()

        if completed_episodes % cfg.algorithm.log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)

            total_num_steps = (
                (n_updates + 1) * cfg.env.parallel_envs * cfg.algorithm.n_steps
            )
            end = time.time()
            logger.info(
                f"Updates {n_updates}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            logger.info(
                f"Last {len(all_infos)} training episodes mean reward {squashed['r'].sum():.3f}"
            )

            all_infos.clear()

        if cfg.algorithm.eval_interval is not None and (
            completed_episodes > last_eval_ep and completed_episodes % cfg.algorithm.eval_interval == 0 or n_updates == num_updates
        ):
            all_infos = evaluate(
				envs,
                exploitation_agent if exploitation_agent is not None else exploration_agent,
                cfg.algorithm.episodes_per_eval,
                cfg.env.device,
            )
            eval_info = _squash_info(all_infos)
            logger.log_evaluation(completed_episodes, eval_info, cfg.algorithm.episodes_per_eval)
            last_eval_ep = completed_episodes

    # save models at very end
    cur_save_dir = os.path.join("models", f"e_{completed_episodes}")
    save_at = cur_save_dir
    os.makedirs(save_at, exist_ok=True)
    exploration_agent.save(save_at)
    if exploitation_agent is not None:
        cur_save_dir = os.path.join("exploitation_models", f"e_{completed_episodes}")
        save_at = cur_save_dir
        os.makedirs(save_at, exist_ok=True)
        exploitation_agent.save(save_at)

    envs.close()

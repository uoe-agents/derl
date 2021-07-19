# Decoupling Exploration and Exploitation in Reinforcement Learning

This repository is the official implementation of **Decoupled Reinforcement Learning (DeRL)**

## Dependencies
Clone and install codebase with relevant dependencies using the provided `setup.py` with
```console
$ git clone git@github.com:uoe-agents/derl.git
$ cd derl
$ pip install -e .
```

We recommend to install dependencies in a virtual environment.



In order to run experiments in the Hallway environment, install using the following code:

```console
$ cd derl/hallway_explore
$ pip install -e .
```

## Training
To train baselines or DeRL algorithms with the identified best hyperparameters, navigate to the derl directory
```console
$ cd derl/derl
```

and execute the script:

```console
$ python run_best.py run --seeds=<NUM_SEEDS> <ENV> <ALG-CONFIG> <INTRINSIC_REWARD> start
```

Valid environments are
- `deepsea_<N>` for N in {10, 14, 20, 24, 30}
- `hallway_<Nl-Nr>` for Nl in {10, 20, 30} and Nr in {N_l, 0}

Valid algorithm configurations can be found in `best_config`:
- `deepsea_a2c`
- `deepsea_ppo`
- `deepsea_dea2c`
- `deepsea_deppo`
- `deepsea_dedqn`
- `hallway_a2c`
- `hallway_ppo`
- `hallway_dea2c`
- `hallway_deppo`
- `hallway_dedqn`

Valid intrinsic rewards for baseline configurations (A2C and PPO) are
- `none`: no intrinsic rewards
- `dict_count`: count-based intrinsic reward with a simple lookup table
- `hash_count`: count-based intrinsic reward with the SimHash hash-function used to group states (<https://arxiv.org/abs/1611.04717>)
- `icm`: prediction-based intrinsic reward of Intrinsic Curiosity Module (ICM) (<https://arxiv.org/abs/1705.05363>)
- `rnd`: prediction-based intrinsic reward of Random Network Distillation (RND) (<https://arxiv.org/abs/1810.12894>)
- `ride`: prediction-based intrinsic reward of Rewarding-Impact-Driven-Exploration (RIDE) (<https://arxiv.org/abs/2002.12292>)

For Decoupled RL algorithms (DeA2C, DePPO, DeDQN), valid intrinsic rewards are
- `dict_count`
- `icm`

## Codebase Structure

### Hydra Configurations
The interface of the main run script `run.py` is handled through [Hydra](https://hydra.cc/) with a hierarchy of configuration files under `configs/`.
These are structured in packages for

- exploration algorithms/ baselines under `configs/algorithm/`
- intrinsic rewards under `configs/curiosity/`
- environments under `configs/env/`
- exploitation algorithms of DeRL under `configs/exploitation_algorithm/`
- hydra parameters under `configs/hydra/`
- logger parameters under `configs/logger/`
- default parameters in `configs/default.yaml`

### On-Policy Algorithms
Two on-policy algorithms are implemented under `on_policy/` which extend the abstract algorithm class found in `on_policy/algorithm.py`:
- Advantage Actor-Critic (A2C) found in `on_policy/algos/a2c.py`
- Proximal Policy Optimisation (PPO) found in `on_policy/algos/ppo.py`

Shared elements such as network models, on-policy storage etc. can be found in `on_policy/common/` and the training script for on-policy algorithms can be found in `on_policy/train.py`.

### Off-Policy Algorithms
For off-policy RL algorithms, only (Double) Deep Q-Networks (DQNs) are implemented under `off_policy/` which extend the abstract algorithm class found in `off_policy/algorithm.py`. The (D)DQN implementation can be found in `off_policy/algos/dqn.py`. Common components such as network models, prioritised and standard replay buffers can be found under `off_policy/common/` and the training script for off-policy algorithms can be found in `off_policy/train.py`.

**DISCLAIMER: Training of off-policy DQN for the exploration policy or baseline is implemented but has not been extensively tested nor evaluated for the paper.**

### Intrinsic Rewards
We consider five different definitions of count- and prediction-based intrinsic rewards for exploration. Their implementations can all be found under `intrinsic_rewards/` and extend the abstract base class found in `intrinsic_rewards/intrinsic_reward.py` which serves as a common interface.

### Utils
Further utilities such as environment wrappers/ setup, loggers and more can be found under `utils/`.


## Citation
```
@inproceedings{schaefer2021decoupling,
	title={Decoupling Exploration and Exploitation in Reinforcement Learning},
	author={Lukas Schäfer and Filippos Christianos and Josiah Hanna and Stefano V. Albrecht},
	booktitle={ICML Workshop on Unsupervised Reinforcement Learning (URL)},
	year={2021}
}
```

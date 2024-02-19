# Synthetic Experimental Framework

This repository includes a customizable simulated environment for testing different aspects of the agent/environment interaction in addressing the behavioral nudging personalization contextual bandit problem. The code was written around 2022 and may be a bit outdated now.

The findings from SXF contributed to the papers
- [Examining Policy Entropy of Reinforcement Learning Agents for Personalization Tasks](https://arxiv.org/abs/2211.11869)
- [Increasing Entropy to Boost Policy Gradient Performance on Personalization Tasks](https://arxiv.org/abs/2310.05324)
- [On the Unreasonable Efficiency of State Space Clustering in Personalization Tasks](https://arxiv.org/abs/2112.13141)

![exp_action_dist](https://github.com/sukiboo/sxf/assets/38059493/931978ba-2d8c-43f4-b3de-01704f24f358)

## Installation
* Install conda / pip requirements via `conda env create -f environment.yml`
* Activate conda environment with `conda activate synthetic-experimental-framework`
* Modify experiment configuration in `configs/config.yml` as needed
* Run a new experiment via `python -m run_experiment`, or load a recorded experiment `exp_name` via `python -m run_experiment --load exp_name`
* Results of the experiment can be found in `exp_data/exp_name/` directory

## File Overview
* `environment.yml` --- list of the required packages
* `configs/config.yml` --- config file containing the experiment/envirnment/agent parameters
* `run_experiment.py` --- main module to run the experiment
* `exp_data/` --- directory containing the experiment data, images, checkpoints
---
* `experiment_component/experiment.py` --- setup the experiment
* `experiment_component/data_visualization.py` --- compute and report results of an experiment
---
* `environment_component/environment.py` --- setup the environment
* `environment_component/state_space.py` --- setup the state space for the environment
* `environment_component/action_space.py` --- setup the action space for the environment
* `environment_component/reward_function.py` --- setup the reward function for the environment
* `environment_component/feedback_signal.py` --- setup the feedback signal that is given to the agent
---
* `agent_component/agent.py` --- setup the agent
* `agent_component/network_architecture.py` --- setup the policy for the agent
* `agent_component/loss_function.py` --- setup the loss function for the agent



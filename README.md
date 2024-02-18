# Synthetic Experimental Framework (v0.1)

## Installation
* Install conda / pip requirements via `conda env create -f environment.yml`
* Activate conda environment with `conda activate lirio-experimental-framework`
* Modify experiment configuration in `config.yml` as needed
* Run a new experiment via `python -m run_experiment`, or load a recorded experiment `exp_name` via `python -m run_experiment --load exp_name`
* Results of the experiment can be found in `exp_data/exp_name/` directory

## File Overview
* `environment.yml` --- list of the required packages
* `config.yml` --- config file containing the experiment/envirnment/agent parameters
* `run_experiment.py` --- main module to run the experiment
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
---
* `exp_data/` --- directory containing the experiment data, images, checkpoints

## TODO list
* implement q-learning algorithms
* implement actor-critic algorithms
* implement MLflow support
* implement static graphs
* try evolutionary algorithms because why not

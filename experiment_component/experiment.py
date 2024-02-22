import os
import time
import yaml
import dill as pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from environment_component.environment import Environment
from agent_component.agent import Agent


class Experiment:
    """Setup the experiment."""

    def __init__(self, params):
        self.__dict__.update(params)
        self.setup_experiment()

    def setup_experiment(self):
        """Configure experiment components."""
        if self.exp_name is None:
            self.exp_name = str(int(time.time()))
        os.makedirs(f'./exp_data/{self.exp_name}', exist_ok=True)
        yaml.dump(self.__dict__, open(f'./exp_data/{self.exp_name}/params.yml', 'w'))
        self.set_random_seed()
        self.stats, self.agents = {'r': {}, 'a': {}, 'l': {}}, []
        self.setup_environment()

    def set_random_seed(self):
        """Fix random seed for reproducibility."""
        self.rng = np.random.default_rng(seed=self.seed)
        self.seed_env = self.rng.integers(1e+09)
        self.seed_agent = self.rng.integers(1e+09)

    def setup_environment(self):
        """Create an environment with specified parameters."""
        self.params_env.update({'seed': self.seed_env})
        self.env = Environment(self.params_env)
        env_r, env_a = self.env.get_env_stats(steps=self.num_steps)
        self.stats['r'].update(env_r)
        self.stats['a'].update(env_a)

    def setup_agent(self, params_agent):
        """Create an agent with specified parameters."""
        params_agent.update({'dim_s': self.env.StateSpace.dim,
                             'actions': self.env.actions,
                             'seed': self.seed_agent})
        agent = Agent(params_agent)
        while agent.name in [a.name for a in self.agents]: agent.name += ' '
        agent.setup_checkpoints(self.exp_name)
        for k in self.stats.keys():
            self.stats[k].update({agent.name: []})
        return agent

    def run(self):
        """Train agents on the environment."""
        print(f'running experiment \'{self.exp_name}\'...')
        for params_agent in self.params_agent:
            agent = self.setup_agent(params_agent)
            self.agents.append(agent)
            self.env.reset_env()
            self.t = 0
            with tqdm(total=self.num_steps, ascii=True, desc=f'{agent.name:>10s} agent') as pbar:
                while self.t < self.num_steps:
                    with tf.GradientTape() as tape:
                        for _ in range(agent.batch_size):
                            self.env_interact(agent)
                            pbar.update(1)
                            if self.t >= self.num_steps:
                                break
                        agent.train_step(tape)
        self.save_exp()

    def env_interact(self, agent):
        """Simulate an agent-environment interaction."""
        self.s = self.env.reset()
        self.a = agent.sample_action(self.s)
        _, self.r, _, _ = self.env.step(self.a)
        agent.buffer.append((self.s, self.a, self.r))
        self.t += 1
        self.record_data(agent)

    def record_data(self, agent):
        """Record an agent-environment interaction data."""
        self.stats['r'][agent.name].append(self.r)
        self.stats['a'][agent.name].append(self.a)
        self.stats['l'][agent.name].append(agent.loss.numpy())
        if self.t % self.ckpt_step == 0:
            agent.checkpoint_policy(self.t)

    def save_exp(self):
        """Save experiment data to a file."""
        save_dir = f'./exp_data/{self.exp_name}'
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/data.pkl', 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)


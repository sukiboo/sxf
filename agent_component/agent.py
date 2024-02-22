import numpy as np
import tensorflow as tf
from collections import deque

from agent_component.network_architecture import NetworkArchitecture
from agent_component.loss_function import LossFunction


class Agent:
    """Create an agent for solving a contextual bandit problem."""

    def __init__(self, params):
        self.__dict__.update(params)
        if 'name' not in params:
            self.name = self.params_arch['arch_type'] + '_' + self.params_loss['loss_type']
        self.create_agent()

    def create_agent(self):
        """Configure and initialize an agent."""
        self.set_random_seed()
        self.setup_network()
        self.setup_learning()

    def set_random_seed(self):
        """Fix random seed for reproducibility."""
        tf.random.set_seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.seed_arch = self.rng.integers(1e+09)
        self.seed_loss = self.rng.integers(1e+09)

    def setup_network(self):
        """Create policy network."""
        self.params_arch.update({'dim_s': self.dim_s, 'actions': self.actions,
                                 'seed': self.seed_arch, 'name': self.name})
        self.policy = NetworkArchitecture(self.params_arch)

    def setup_learning(self, buffer_size=100000):
        """Create loss function."""
        self.params_loss.update({'seed': self.seed_loss})
        self.buffer = deque([], maxlen=buffer_size)
        self.LossFunction = LossFunction(self.params_loss)
        self.loss = tf.constant(0, dtype='float32')
        self.optimizer = self.LossFunction.setup_optimizer()

    def setup_checkpoints(self, exp_name):
        """Configure checkpoints to save agent's policy."""
        self.checkpoint = tf.train.Checkpoint(policy=self.policy)
        self.setup_manager(exp_name)
        self.checkpoint_policy(0)

    def setup_manager(self, exp_name):
        """Configure checkpoint manager system."""
        checkpoint_dir = f'./exp_data/{exp_name}/checkpoints/{self.name}'
        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
            directory=checkpoint_dir, checkpoint_name=self.name, max_to_keep=None)

    def checkpoint_policy(self, t):
        """Save agent's policy weights."""
        self.manager.save(checkpoint_number=t)

    def compute_loss(self):
        """Compute loss function."""
        loss = self.LossFunction.total_loss(self)
        return loss

    def train_step(self, tape):
        """Perform an iteration of training algorithm."""
        self.loss = self.compute_loss()
        self.grads = tape.gradient(self.loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(self.grads, self.policy.trainable_variables))

    def get_action_probs(self, s):
        """Compute probability distribution over the action space."""
        action_probs = tf.nn.softmax(self.temperature * self.policy(s))
        return action_probs

    def sample_action(self, s):
        """Sample an action for the given state."""
        logits = self.policy(s)
        action = tf.random.categorical(self.temperature * logits, num_samples=1).numpy().item()
        return action


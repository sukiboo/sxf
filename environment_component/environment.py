import gym
import numpy as np

from environment_component.state_space import StateSpace
from environment_component.action_space import ActionSpace
from environment_component.feedback_signal import FeedbackSignal
from environment_component.reward_function import RewardFunction

gym.logger.set_level(40)


class Environment(gym.Env):
    """Generate synthetic contextual bandit environment."""

    def __init__(self, params):
        super().__init__()
        self.__dict__.update(params)
        self.set_random_seed()
        self.reset_env()

    def set_random_seed(self):
        """Fix random seed for reproducibility."""
        self.rng = np.random.default_rng(seed=self.seed)
        self.seed_state = self.rng.integers(1e+09)
        self.seed_action = self.rng.integers(1e+09)
        self.seed_feedback = self.rng.integers(1e+09)

    def reset_env(self):
        """Setup the environment."""
        self.setup_state_space()
        self.setup_action_space()
        self.setup_feedback_signal()
        self.setup_reward_function()

    def setup_state_space(self):
        """Generate state space."""
        self.params_state.update({'seed': self.seed_state})
        self.StateSpace = StateSpace(self.params_state)
        self.observe = self.StateSpace.observe_state
        self.observation_space = gym.spaces.Box(low=self.StateSpace.low, high=self.StateSpace.high,
                                                shape=(self.StateSpace.dim,), dtype=np.float32)

    def setup_action_space(self):
        """Generate action space."""
        self.params_action.update({'seed': self.seed_action})
        self.ActionSpace = ActionSpace(self.params_action)
        self.actions = self.ActionSpace.generate_available_actions()
        self.action_space = gym.spaces.Discrete(self.ActionSpace.num)

    def setup_feedback_signal(self):
        """Generate feedback signal."""
        self.params_feedback.update({'dim_s': self.StateSpace.dim,
                                     'dim_a': self.ActionSpace.dim,
                                     'seed': self.seed_feedback})
        self.FeedbackSignal = FeedbackSignal(self.params_feedback)
        self.get_feedback = self.FeedbackSignal.get_feedback

    def setup_reward_function(self):
        """Configure reward values that agent receives."""
        self.params_reward.update({'actions': self.actions,
                                   'num_a': self.ActionSpace.num,
                                   'get_feedback': self.get_feedback})
        self.RewardFunction = RewardFunction(self.params_reward)
        self.compute_reward = self.RewardFunction.compute_reward

    def reset(self):
        """Observe a new state."""
        self.state = self.observe().flatten()
        return self.state

    def step(self, action_index):
        """Given an observed state take an action and receive reward."""
        reward = self.compute_reward(self.state, action_index).item()
        done = True
        info = {}
        return self.state, reward, done, info

    def get_env_stats(self, steps):
        """Compute the average/minimum/maximum reward values and optimal actions."""
        self.reset_env()
        S = self.observe(num=steps)
        r_vals = self.compute_reward(S)
        self.stats_r = {'avg': r_vals.mean(axis=1).tolist(),
                        'min': r_vals.min(axis=1).tolist(),
                        'max': r_vals.max(axis=1).tolist()}
        self.stats_a = {'env': r_vals.argmax(axis=1).tolist()}
        return self.stats_r, self.stats_a



import numpy as np


class RewardFunction:
    '''generate reward values given to the agent'''

    def __init__(self, params):
        self.__dict__.update(params)
        self.configure_reward()

    def configure_reward(self):
        '''configure reward function'''
        if self.reward_type == 'scale':
            self.scale = self.params_cont['scale']
            self.compute_reward = self.compute_reward_scale
        elif self.reward_type == 'interval':
            self.r_min, self.r_max = np.sort(self.params_cont['interval'])
            self.compute_reward = self.compute_reward_interval
        elif self.reward_type == 'discrete':
            self.__dict__.update(self.params_disc)
            self.r_vals = np.sort(self.values + [self.default] * (self.num_a - len(self.values)))
            self.compute_reward = self.compute_reward_disc
        else:
            raise ValueError(f'reward type \'{self.reward_type}\' is not implemented...')

    def compute_reward_scale(self, s, action_index=None):
        '''compute continuous reward function by scaling the feedback'''
        a = self.actions[action_index] if action_index is not None else self.actions
        f = self.get_feedback(s,a)
        r = self.scale * f
        return r

    def compute_reward_interval(self, s, action_index=None):
        '''compute continuous reward values by mapping feedback signal to the interval'''
        F = self.get_feedback(s, self.actions)
        f_min = F.min(axis=1, keepdims=True)
        f_max = F.max(axis=1, keepdims=True)
        R = ((self.r_max - self.r_min) * F + self.r_min * f_max - self.r_max * f_min) / (f_max - f_min)
        r = R[0,action_index] if action_index is not None else R
        return r

    def compute_reward_disc(self, s, action_index=None):
        '''compute discrete reward values'''
        F = self.get_feedback(s, self.actions)
        R = self.r_vals[np.argsort(F, axis=1)]
        r = R[0,action_index] if action_index is not None else R
        return r


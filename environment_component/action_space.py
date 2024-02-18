
import numpy as np


class ActionSpace:
    '''setup action space and sample available actions'''

    def __init__(self, params):
        self.__dict__.update(params)
        self.rng = np.random.default_rng(seed=self.seed)

    def generate_available_actions(self, num=None):
        '''generate the set of available actions'''
        if num is None:
            num = self.num
        if self.dist == 'uniform':
            self.actions = self.rng.uniform(self.low, self.high, (num,self.dim))
        elif self.dist == 'gauss':
            self.actions = np.clip(self.rng.standard_normal((num,self.dim)), self.low, self.high)
        else:
            raise ValueError(f"action distribution '{self.dist}' is not implemented...")
        return self.actions.astype(np.float32)


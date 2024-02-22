import numpy as np


class StateSpace:
    """Setup state space and observation generator."""

    def __init__(self, params_state):
        self.__dict__.update(params_state)
        self.rng = np.random.default_rng(seed=self.seed)

    def observe_state(self, num=1):
        """Generate observed states."""
        if self.dist == 'uniform':
            self.state = self.rng.uniform(self.low, self.high, (num,self.dim))
        elif self.dist == 'gauss':
            self.state = np.clip(self.rng.standard_normal((num,self.dim)), self.low, self.high)
        else:
            raise ValueError(f"state distribution '{self.dist}' is not implemented...")
        return self.state.astype(np.float32)


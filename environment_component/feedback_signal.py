import numpy as np


class SyntheticGaussianMapping:
    """Generate synthetic feature extractor."""

    def __init__(self, params):
        self.__dict__.update(params)
        self.activation = lambda z: np.exp(-z**2)
        self.rng = np.random.default_rng(seed=self.seed)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the network from the normal distribution."""
        self.dims = [self.dim_in, *self.dim_layers, self.dim_out]
        self.num_layers = len(self.dims) - 1
        self.weights = {}
        for l in range(self.num_layers):
            self.weights[l] = self.rng.normal(scale=1., size=(self.dims[l]+1,self.dims[l+1]))

    def propagate(self, x):
        """Propagate input through the network."""
        z = np.array(x, ndmin=2)
        for l in range(self.num_layers):
            z = np.concatenate([np.ones((z.shape[0],1)), z], axis=1)
            if l < self.num_layers - 1:
                z = self.activation(np.matmul(z, self.weights[l]))
            else:
                z = np.tanh(np.matmul(z, self.weights[l]))
        return z


class FeedbackSignal:
    """Generate synthetic feedback signal."""

    def __init__(self, params):
        self.__dict__.update(params)
        self.rng = np.random.default_rng(seed=self.seed)
        self.generate_state_embedding()
        self.generate_action_embedding()

    def generate_state_embedding(self):
        """Generate state feature map."""
        self.seed_s = self.rng.integers(1e+09)
        self.params_s = {'dim_in': self.dim_s, 'dim_layers': self.arch_s,
                         'dim_out': self.dim_feature, 'seed': self.seed_s}
        self.feature_map_s = SyntheticGaussianMapping(self.params_s)
        self.feature_s = lambda s: self.feature_map_s.propagate(s)

    def generate_action_embedding(self):
        """Generate action feature map."""
        self.seed_a = self.rng.integers(1e+09)
        self.params_a = {'dim_in': self.dim_a, 'dim_layers': self.arch_a,
                         'dim_out': self.dim_feature, 'seed': self.seed_a}
        self.feature_map_a = SyntheticGaussianMapping(self.params_a)
        self.feature_a = lambda a: self.feature_map_a.propagate(a)

    def feature_relevance(self, s, a):
        """Measure state and action relevance in the latent feature space."""
        if self.relevance == 'cossim':
            norm_s = np.linalg.norm(s, axis=1, keepdims=True)
            norm_a = np.linalg.norm(a, axis=1, keepdims=True)
            rel = np.matmul(s, a.T) / np.matmul(norm_s, norm_a.T)
        elif self.relevance == 'inner':
            rel = np.matmul(s, a.T)
        else:
            raise ValueError(f"feature relevance function '{self.relevance}' is not implemented...")
        return rel

    def get_feedback(self, s, a):
        """Compute generated feedback signal on a given state-action pair."""
        self.feedback = self.feature_relevance(self.feature_s(s), self.feature_a(a))
        return self.feedback


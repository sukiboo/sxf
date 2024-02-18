
import tensorflow as tf
from tensorflow.keras.layers import Dense


class NetworkArchitecture(tf.keras.Model):
    '''create policy network of specified architecture'''

    def __init__(self, params):
        super().__init__()
        self.__dict__.update(params)
        self.num_a, self.dim_a = self.actions.shape
        self._name = params['name']
        self.configure_network()

    def configure_network(self):
        '''initialize and build the model'''
        tf.random.set_seed(self.seed)
        if self.arch_type == 'ffnn':
            self.create_network_ffnn()
            self.call = self.call_ffnn
        elif self.arch_type == 'drrn':
            self.create_network_drrn()
            self.call = self.call_drrn
        else:
            raise ValueError(f'network architecture \'{self.arch_type}\' is not implemented...')
        self.build(input_shape=(None,self.dim_s))

    def create_network_ffnn(self):
        '''create network with feed-forward architecture'''
        self.layers_s = []
        for nodes in self.arch:
            self.layers_s.append(Dense(nodes, activation=self.activation, name='state_layer'))
        self.layer_out = Dense(self.num_a, activation=None, name='output_layer')

    def create_network_drrn(self):
        '''create network with DRRN architecture'''
        self.actions_tf = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        self.layers_s, self.layers_a = [], []
        for nodes in self.arch:
            self.layers_s.append(Dense(nodes, activation=self.activation, name='state_layer'))
            self.layers_a.append(Dense(nodes, activation=self.activation, name='action_layer'))

    def state_branch(self, s):
        '''define state feature extractor'''
        out = tf.reshape(tf.convert_to_tensor(s, dtype=tf.float32), shape=[-1,self.dim_s])
        for layer in self.layers_s:
            out = layer(out)
        return out

    def action_branch(self, a):
        '''define action feature extractor'''
        if self.arch_type == 'ffnn':
            pass
        elif self.arch_type == 'drrn':
            out = tf.reshape(tf.convert_to_tensor(a, dtype=tf.float32), shape=[-1,self.dim_a])
            for layer in self.layers_a:
                out = layer(out)
            return out

    def call_ffnn(self, s, training=False):
        '''forward pass through the created feed-forward network'''
        out = self.state_branch(s)
        out = self.layer_out(out)
        return out

    def call_drrn(self, s, a=None, training=False):
        '''forward pass through the created DRRN network'''
        out_s = self.state_branch(s)
        if a is not None:
            out_a = self.action_branch(a)
        else:
            out_a = self.action_branch(self.actions)
        if self.relevance == 'inner':
            out = tf.matmul(out_s, out_a, transpose_b=True)
        elif self.relevance == 'cossim':
            norm_s = tf.norm(out_s, axis=1, keepdims=True)
            norm_a = tf.norm(out_a, axis=1, keepdims=True)
            out = tf.matmul(out_s / norm_s, out_a / norm_a, transpose_b=True)
        elif self.relevance == 'a_norm':
            norm_a = tf.norm(out_a, axis=1, keepdims=True)
            out = tf.matmul(out_s, out_a / norm_a, transpose_b=True)
        elif self.relevance == 's_norm':
            norm_s = tf.norm(out_s, axis=1, keepdims=True)
            out = tf.matmul(out_s / norm_s, out_a, transpose_b=True)
        return out



import numpy as np
import tensorflow as tf


class LossFunction:
    '''create loss function and optimizer for a given learning objective'''

    def __init__(self, params):
        self.__dict__.update(params)
        self.configure_loss()
        self.configure_regularization()
        self.total_loss = lambda agent: self.compute_loss(agent) + self.compute_reg(agent)

    def configure_loss(self):
        '''setup the loss function'''
        np.random.seed(self.seed)
        if self.loss_type == 'pg':
            self.compute_loss = self.compute_loss_pg
        elif self.loss_type == 'q':
            self.compute_loss = self.compute_loss_q
        else:
            raise ValueError(f'learning type \'{self.loss_type}\' is not implemented...')

    def compute_loss_pg(self, agent):
        '''policy gradient loss function'''
        S, A, R = zip(*[agent.buffer[i] for i in range(-agent.batch_size,0)])
        logits = agent.policy(S)
        probs = tf.nn.softmax(agent.temperature * logits, axis=1)
        logprobs = tf.gather_nd(tf.math.log(probs), list(zip(range(len(A)), A)))
        loss = -tf.reduce_mean(R * logprobs)
        return loss

    def compute_loss_q(self, agent):
        '''q-learning loss function'''
        batch_size = min(len(agent.buffer), agent.batch_size)
        ind = np.random.choice(len(agent.buffer), size=batch_size, replace=False)
        S, A, R = zip(*[agent.buffer[i] for i in ind])
        out = agent.policy(S)
        Q = tf.gather_nd(out, list(zip(range(len(A)), A)))
        loss = tf.keras.metrics.mean_squared_error(R, Q)
        return loss

    def configure_regularization(self):
        '''add regularization to the loss function'''
        regs = []
        if hasattr(self, 'regularization'):
            if self.regularization.get('entropy_batch', 0) != 0:
                regs.append(self.compute_entropy_reg_batch)
            if self.regularization.get('entropy', 0) != 0:
                regs.append(self.compute_entropy_reg)
            if self.regularization.get('l2', 0) != 0:
                regs.append(self.compute_l2_reg)
        self.compute_reg = lambda agent: sum(reg(agent) for reg in regs)

    def compute_entropy_reg(self, agent):
        '''compute the entropy of the agent's probability distribution over the action space'''
        S = [t[0] for t in list(agent.buffer)[-agent.batch_size:]]
        logits = agent.policy(S)
        probs = tf.nn.softmax(agent.temperature * logits, axis=1)
        entropy = -tf.reduce_sum([prob * tf.math.log(prob + 1e-08) for prob in probs]) / agent.batch_size
        reg = -self.regularization['entropy'] * entropy
        return reg

    def compute_entropy_reg_batch(self, agent):
        '''compute the entropy on the batch of transitions'''
        S = [t[0] for t in list(agent.buffer)[-1000:]]
        prob = tf.reduce_mean(tf.nn.softmax(agent.temperature * agent.policy(S)), axis=0)
        entropy = -tf.reduce_sum(prob * tf.math.log(prob + 1e-08))
        reg = -self.regularization['entropy_batch'] * entropy
        return reg

    def compute_l2_reg(self, agent):
        '''compute the l2-norm of the agent's weights'''
        weight_norm = tf.reduce_sum([tf.norm(w) for w in agent.policy.weights[::2]])
        reg = self.regularization['l2'] * weight_norm
        return reg

    def setup_optimizer(self):
        '''setup optimization algorithm for the loss function'''
        optimizer = getattr(tf.keras.optimizers, self.opt_alg['name'])(**self.opt_alg)
        return optimizer


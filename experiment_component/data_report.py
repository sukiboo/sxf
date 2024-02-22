import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from experiment_component.data_visualization import *

np.set_printoptions(precision=3, suppress=True)


class DataReport:
    """Compute and report the outcome of an experiment."""

    def __init__(self, exp):
        self.exp = exp
        self.img_dir = f'./exp_data/{exp.exp_name}/images'
        os.makedirs(self.img_dir, exist_ok=True)
        self.evaluate_agents()
        self.compute_metrics()

    def evaluate_agents(self, num_s=10000):
        """Deterministically evaluate agents on the environment."""
        self.exp.env.reset_env()
        S = self.exp.env.observe(num=num_s)
        R = self.exp.env.compute_reward(S)
        r_mean, r_max = R.mean(axis=1), R.max(axis=1)
        eval_str = 'agents evaluation:'
        print(eval_str)
        for agent in self.exp.agents:
            A = agent.policy(S).numpy().argmax(axis=1)
            r = (R[range(num_s),A] - r_mean) / (r_max - r_mean)
            hist = np.histogram(A, bins=np.arange(len(self.exp.env.actions) + 1), density=True)[0]
            perf_str = f'{agent.name:>8s} -- {r.mean():.4f}'
            hist_str = ' '.join(f'{p:.3f}' for p in -np.sort(-hist)[:10])
            print(f'  {perf_str},  histogram: [{hist_str}]')
            eval_str += f'\n{perf_str}, [{hist_str}]'
        with open(f'{self.img_dir}/eval.txt', 'w+') as eval_file:
            eval_file.write(eval_str)

    def compute_metrics(self):
        """Compute various metrics."""
        print('computing metrics...')
        self.reward, self.reward_norm = self.get_reward()
        self.actions = self.get_actions()
        self.loss = self.get_loss()
        self.dist = self.get_action_dist(num_s=1000)
        self.emb_s = self.get_state_embeddings(num_s=1000)
        self.emb_a = self.get_action_embeddings()

    def report(self):
        """Report and visualize various metrics."""
        plot_actions(self)
        plot_loss(self, smoothing=100)
        plot_reward(self, smoothing=5000, norm=True)
        plot_action_histogram(self)
        plot_state_embeddings_cossim(self)
        plot_action_embeddings_cossim(self)

    def full_report(self):
        """Report and visualize various metrics."""
        self.report()
        plot_action_dist_gif(self, num_s=100)
        plot_embeddings_gif(self, method='pca', num_s=200)
        plot_weights_gif(self)

    def get_reward(self):
        """Get reward and normalized reward values."""
        R = pd.DataFrame(self.exp.stats['r'])
        R_n = R.sub(R['avg'], axis=0)
        R_n = R_n.div(R_n['max'], axis=0)
        R_n = R_n.drop(['avg', 'min', 'max'], axis=1)
        return R, R_n

    def get_actions(self):
        """Get selected actions."""
        A = pd.DataFrame(self.exp.stats['a'])
        return A

    def get_loss(self):
        """Get loss values."""
        L = pd.DataFrame(self.exp.stats['l'])
        return L

    def get_action_dist(self, num_s):
        """Compute average probability distributions over the action space."""
        self.exp.env.reset_env()
        S = self.exp.env.observe(num=num_s)
        R = self.exp.env.compute_reward(S)
        ##temperature = 10 / (R.max(axis=1) - R.mean(axis=1)).mean()
        temperature = 10
        dist = {agent.name: [] for agent in self.exp.agents}
        for agent in self.exp.agents:
            for checkpoint in agent.manager.checkpoints:
                agent.checkpoint.restore(checkpoint)
                dist[agent.name].append(agent.get_action_probs(S).numpy())
        dist_env = np.exp(temperature*R) / np.exp(temperature*R).sum(axis=1, keepdims=True)
        dist['env'] = [dist_env] * (1 + self.exp.num_steps // self.exp.ckpt_step)
        return dist

    def get_weights(self):
        """Get agents' weights."""
        weights = {agent.name: [] for agent in self.exp.agents}
        for agent in self.exp.agents:
            for checkpoint in agent.manager.checkpoints:
                agent.checkpoint.restore(checkpoint)
                weights[agent.name].append(agent.policy.get_weights())
        return weights

    def get_state_embeddings(self, num_s=None):
        """Compute state embeddings."""
        if num_s is not None:
            self.exp.env.reset_env()
            S = self.exp.env.observe(num=num_s)
        else:
            dim = self.exp.env.params_state['dim']
            low = self.exp.env.params_state['low']
            high = self.exp.env.params_state['high']
            S_low = (low + high) / 2 * np.ones((dim,dim)) + (low - high) / 2 * np.eye(dim)
            S_high = (low + high) / 2 * np.ones((dim,dim)) + (high - low) / 2 * np.eye(dim)
            S = np.concatenate((S_low, S_high), axis=0)
        emb = {agent.name: [] for agent in self.exp.agents}
        for agent in self.exp.agents:
            for checkpoint in agent.manager.checkpoints:
                agent.checkpoint.restore(checkpoint)
                emb[agent.name].append(agent.policy.state_branch(S).numpy())
        emb['env'] = [self.exp.env.FeedbackSignal.feature_s(S)]
        return emb

    def get_action_embeddings(self, num_a=None):
        """Compute action embeddings."""
        if num_a is not None:
            self.exp.env.reset_env()
            A = self.exp.env.ActionSpace.generate_available_actions(num=num_a)
        else:
            A = self.exp.env.actions
        emb = {}
        for agent in self.exp.agents:
            if agent.policy.arch_type == 'drrn':
                emb.update({agent.name: []})
                for checkpoint in agent.manager.checkpoints:
                    agent.checkpoint.restore(checkpoint)
                    emb[agent.name].append(agent.policy.action_branch(A).numpy())
        emb['env'] = [self.exp.env.FeedbackSignal.feature_a(A)]
        return emb

    def get_embeddings(self, method='pca', num_s=None):
        """Compute embeddings on the state and action branches."""
        if method == 'pca':
            proj = lambda x: PCA(n_components=2).fit_transform(x)
        elif method == 'tsne':
            proj = lambda x: TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(x)
        else:
            raise ValueError(f"dimensionality reduction method '{method}' is not implemented...")
        emb_s = self.get_state_embeddings(num_s=num_s)
        emb = {}
        for name in self.emb_s.keys():
            try:
                data = [*emb_s[name], *self.emb_a[name]]
            except:
                data = [*emb_s[name]]
            ind = np.cumsum(list(map(lambda x: x.shape[0], data)))[:-1]
            emb[name] = np.split(proj(np.concatenate(data, axis=0)), ind, axis=0)
        return emb


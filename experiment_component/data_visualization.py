import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import seaborn as sns

sbn_bold = ['#2288dd', '#dd8822', '#22dd88', '#dd2288', '#8822dd', '#88dd22']
sbn_mute = ['#66aadd', '#ddaa66', '#66ddaa', '#dd66aa', '#aa66dd', '#aadd66']
sbn_base = np.array([sbn_bold, sbn_mute]).flatten(order='C')
sbn_pair = np.array([sbn_bold, sbn_mute]).flatten(order='F')
sns.set_theme(style='darkgrid', palette=sbn_base, font='monospace')


def plot_reward(data, smoothing, norm=True):
    """Plot received reward values."""
    R = data.reward_norm if norm else data.reward
    R = pd.concat([pd.DataFrame(0, index=range(-smoothing,0), columns=R.columns), R])
    R = R.rolling(smoothing, min_periods=smoothing).mean()
    R.index += 1
    sns.set_palette(sbn_base)
    R.plot(figsize=(10,6), linewidth=3, alpha=.75)
    plt.xlabel('timesteps')
    plt.ylabel('normalized return')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=R.columns.size, prop={'size': 16})
    plt.tight_layout()
    plt.savefig(f'{data.img_dir}/reward.pdf', format='pdf')
    plt.close()

def plot_loss(data, smoothing):
    """Plot observed loss values."""
    L = pd.concat([pd.DataFrame(data.loss.iloc[[0]*smoothing]), data.loss])
    L = L.rolling(smoothing, min_periods=smoothing).mean()
    L.index += 1
    sns.set_palette(sbn_base)
    L.plot(figsize=(10,6), linewidth=3, alpha=.75)
    plt.ylim(-1,1)
    plt.yscale('symlog')
    plt.xlabel('timesteps')
    plt.ylabel('loss value')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{data.img_dir}/loss.pdf', format='pdf')
    plt.close()

def plot_actions(data):
    """Plot indices of the actions taken."""
    actions = data.actions.copy()
    actions.insert(actions.columns.size-1, 'optimal', actions.pop('env'))
    sns.set_palette(sbn_base)
    fig = plt.figure(figsize=(10,6))
    plt.hist(actions, bins=np.arange(data.exp.env.ActionSpace.num + 1), rwidth=1, density=True)
    plt.xlabel('action space')
    plt.ylabel('frequency percentage')
    plt.xlim(0, data.exp.env.ActionSpace.num)
    plt.ylim(0, 10/data.exp.env.ActionSpace.num)
    plt.legend(actions.columns, loc='upper center', bbox_to_anchor=(.5,1.1), ncol=actions.columns.size)
    plt.tight_layout()
    plt.savefig(f'{data.img_dir}/actions.pdf', format='pdf')
    plt.close()

def plot_state_embeddings_cossim(data, low=10, high=90):
    """Plot absolute values of cosine similarity of state embeddings."""
    emb_cossim, emb_cossim_low, emb_cossim_high = {}, {}, {}
    for name in data.emb_s.keys():
        emb_cossim.update({name: []})
        emb_cossim_low.update({name: []})
        emb_cossim_high.update({name: []})
        for emb in data.emb_s[name]:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-08
            cossim = np.abs(np.matmul(emb, emb.T))
            emb_cossim[name].append(np.mean(cossim))
            emb_cossim_low[name].append(np.percentile(cossim,low))
            emb_cossim_high[name].append(np.percentile(cossim,high))
    emb_cossim['env'] *= 1 + data.exp.num_steps // data.exp.ckpt_step
    E = pd.DataFrame(emb_cossim)
    E.index *= data.exp.ckpt_step
    sns.set_palette(sbn_base)
    ax = E.plot(figsize=(10,6), linewidth=3, alpha=.8, legend=True)
    for name in data.emb_s.keys():
        ax.fill_between(E.index, emb_cossim_low[name], emb_cossim_high[name], alpha=.2)
    plt.xlabel('timesteps')
    plt.ylabel('cosine similarity of state embeddings')
    plt.tight_layout()
    plt.savefig(f'{data.img_dir}/state_emb_cossim.pdf', format='pdf')
    plt.close()

def plot_action_embeddings_cossim(data, low=10, high=90):
    """Plot absolute values of cosine similarity of action embeddings."""
    emb_cossim, emb_cossim_low, emb_cossim_high = {}, {}, {}
    for name in data.emb_a.keys():
        emb_cossim.update({name: []})
        emb_cossim_low.update({name: []})
        emb_cossim_high.update({name: []})
        for emb in data.emb_a[name]:
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-08
            cossim = np.abs(np.matmul(emb, emb.T))
            emb_cossim[name].append(np.mean(cossim))
            emb_cossim_low[name].append(np.percentile(cossim,low))
            emb_cossim_high[name].append(np.percentile(cossim,high))
    emb_cossim['env'] *= 1 + data.exp.num_steps // data.exp.ckpt_step
    E = pd.DataFrame(emb_cossim)
    E.index *= data.exp.ckpt_step
    sns.set_palette(sbn_base)
    ax = E.plot(figsize=(10,6), linewidth=3, alpha=.8, legend=True)
    for name in data.emb_a.keys():
        ax.fill_between(E.index, emb_cossim_low[name], emb_cossim_high[name], alpha=.2)
    plt.xlabel('timesteps')
    plt.ylabel('cosine similarity of action embeddings')
    plt.tight_layout()
    plt.savefig(f'{data.img_dir}/action_emb_cossim.pdf', format='pdf')
    plt.close()

def plot_action_dist_gif(data, fps=5, num_s=100):
    """Plot computed probability distributions as a gif."""
    dist = data.get_action_dist(num_s=num_s)
    num_plots = 1 + len(data.exp.agents)
    fig = plt.figure(figsize=(6*num_plots,6))
    axs = [fig.add_axes([.04 + .9*n/num_plots, .03, .9/num_plots - .05, .9]) for n in range(num_plots)]
    norm = mpl.colors.Normalize(0,1)
    im = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn')
    aspect = dist['env'][0].shape[1] / dist['env'][0].shape[0]
    def plot_dist_frame(t):
        '''plot a single frame'''
        fig.suptitle(f'Probability distribution after {t*data.exp.ckpt_step} steps', fontweight='bold')
        for k in range(len(dist)):
            name = list(dist.keys())[k]
            axs[k].matshow(dist[name][t], aspect=aspect, norm=norm, cmap='RdYlGn')
            axs[k].set_title(name, fontweight='bold')
        for ax in axs:
            ax.set_xlabel('action space')
            ax.set_xticks([])
            ax.xaxis.set_ticks_position('bottom')
            ax.set_ylabel('observed states')
            ax.set_yticks([])
    num_frames = 1 + data.exp.num_steps // data.exp.ckpt_step
    dist_gif = anim.FuncAnimation(fig, plot_dist_frame, frames=num_frames)
    fig.colorbar(im, cax=fig.add_axes([.93,.105,.025,.75]))
    dist_gif.save(f'{data.img_dir}/action_dist.gif', fps=fps, dpi=100)
    plt.close()

def plot_weights_gif(data, fps=10):
    """Plot agents' weights as gifs."""
    data.weights = data.get_weights()
    for agent in data.exp.agents:
        weight = data.weights[agent.name]
        num_plots = len(weight[0]) // 2
        fig = plt.figure(figsize=(6*num_plots,6))
        axs = [fig.add_axes([.04 + .9*n/num_plots, .03, .9/num_plots - .05, .9]) for n in range(num_plots)]
        norm = mpl.colors.Normalize(-0.25,0.25)
        im = mpl.cm.ScalarMappable(norm=norm, cmap='bwr')
        def plot_weight_frame(t):
            '''plot a single frame'''
            fig.suptitle(f'{agent.name} weights after {t*data.exp.ckpt_step} steps', fontweight='bold')
            for k in range(num_plots):
                aspect = weight[t][2*k].shape[1] / weight[t][2*k].shape[0]
                axs[k].matshow(weight[t][2*k], aspect=aspect, norm=norm, cmap='bwr')
                axs[k].set_title(f'layer {k+1} weights', fontweight='bold')
                axs[k].xaxis.set_ticks_position('bottom')
        weight_gif = anim.FuncAnimation(fig, plot_weight_frame, frames=len(weight))
        fig.colorbar(im, cax=fig.add_axes([.93,.105,.025,.75]))
        weight_gif.save(f'{data.img_dir}/weights_{agent.name}.gif', fps=fps, dpi=100)
        plt.close()

def plot_embeddings_gif(data, fps=5, method='pca', num_s=None):
    """Plot embedding of state and action embeddings as a gif."""
    data.emb = data.get_embeddings(method=method, num_s=num_s)
    num_plots = 1 + len(data.exp.agents)
    num_frames = 1 + data.exp.num_steps // data.exp.ckpt_step
    fig = plt.figure(figsize=(6*num_plots,6))
    axs = [fig.add_axes([.02+n/num_plots, .06, 1/num_plots-.03, .84]) for n in range(num_plots)]

    def plot_dist_frame(t):
        """Plot a single frame."""
        for ax in axs:
            ax.cla()
        fig.suptitle(f'State embeddings after {t*data.exp.ckpt_step} steps', fontweight='bold')
        axs[0].scatter(data.emb['env'][0][:,0], data.emb['env'][0][:,1])
        axs[0].scatter(data.emb['env'][1][:,0], data.emb['env'][1][:,1])
        axs[0].set_title('environment', fontweight='bold')
        for k in range(len(data.exp.agents)):
            name = data.exp.agents[k].name
            axs[k+1].scatter(data.emb[name][t][:,0], data.emb[name][t][:,1])
            if len(data.emb[name]) > num_frames:
                axs[k+1].scatter(data.emb[name][t+num_frames][:,0], data.emb[name][t+num_frames][:,1])
            axs[k+1].set_title(name, fontweight='bold')

    emb_gif = anim.FuncAnimation(fig, plot_dist_frame, frames=num_frames)
    emb_gif.save(f'{data.img_dir}/embeddings_{method}.gif', fps=fps, dpi=100)
    plt.close()

def plot_action_histogram(data):
    """Plot agents' action selection histograms throughout the training process."""
    hist_index = np.arange(1 + data.exp.num_steps // data.exp.ckpt_step) * data.exp.ckpt_step
    for name in data.dist:
        sns.set_palette('Paired')
        fig, ax = plt.subplots(figsize=(10,6))
        df = pd.DataFrame([dist.mean(axis=0) for dist in data.dist[name]], index=hist_index)
        df.plot.bar(stacked=True, width=1, ax=ax, linewidth=.1, legend=None)
        plt.xticks(np.linspace(0, len(df) - 1, 11), rotation=0)
        ax.set_ylim(0,1)
        ax.set_xlabel('timesteps')
        ax.set_ylabel(f'action distribution')
        ax.set_title(f'{name} histogram', weight='bold', size='xx-large',
                     color=sbn_base[list(data.dist.keys()).index(name)])
        plt.tight_layout()
        plt.savefig(f'./{data.img_dir}/histogram_{name}.png', dpi=200, format='png')
        plt.close()


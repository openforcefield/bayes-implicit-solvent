import matplotlib.pyplot as plt
#from bayes_implicit_solvent.marginal_likelihood.single_type_forward_smc import ESS
from bayes_implicit_solvent.utils import remove_top_right_spines
import numpy as np

from scipy.special import logsumexp


def ESS(log_weights):
    """
    TODO: look also at the function whose expectation we're trying to approximate...

    See "Rethinking the effective sample size" https://arxiv.org/abs/1809.04129
    and references therein for some inspiration in this direction...
    """
    log_Z = logsumexp(log_weights)
    weights = np.exp(log_weights - log_Z)
    return 1 / np.sum(weights ** 2)


def ax_setup(ax=None):
    if type(ax) == type(None):
        ax = plt.subplot()
    remove_top_right_spines(ax)
    return ax

class Plotter():
    def __init__(self, path_to_result):
        self.result = np.load(path_to_result)
        self.particle_snapshots = self.result['particle_snapshots']
        self.current_log_weights = self.result['current_log_weights']
        self.lambdas = self.result['lambdas']

    def scatter(self, t=1, global_ax_lock=True, ax=None):
        particle_snapshots = self.particle_snapshots
        ax = ax_setup(ax)

        T = t
        if global_ax_lock: T = 0
        min_x, min_y = np.min(particle_snapshots[T:], axis=(0, 1))[:2] * 0.95
        max_x, max_y = np.max(particle_snapshots[T:], axis=(0, 1))[:2] / 0.95
        x, y = particle_snapshots[t].T[:2]

        # TODO: Revert to ax.scatter!
        plt.scatter(x, y, c=self.current_log_weights[t]);
        ax.set_xlabel(r'$\theta_0$')
        ax.set_ylabel(r'$\theta_1$')
        # TODO: Figure out how to bind it to the scatter plot!
        cbar = plt.colorbar(shrink=0.5, orientation='horizontal');
        cbar.set_label('log importance weight')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_title('importance-weighted parameter samples')

    def plot_lambdas(self, t=1, ax=None):

        ax = ax_setup(ax)

        ax.plot(self.lambdas[:t])
        ax.set_xlim(0, len(self.lambdas))
        ax.set_yscale('log')
        ax.set_ylim(self.lambdas[1], 1)
        ax.set_xlabel('iteration $i$')
        ax.set_ylabel(r'$\lambda_i$')
        ax.set_title('on-the-fly protocol')


    def plot_work_means(self, t=1, ax=None):
        ax = ax_setup(ax)

        mean_works = -np.mean(self.current_log_weights, 1)[:t]
        ax.plot(mean_works)


    def plot_ESS_traj(self, t=1, ax=None):
        ax = ax_setup(ax)
        ESS_traj = np.array(list(map(ESS, self.current_log_weights[:t])))
        ax.plot(ESS_traj)
        ax.set_xlim(0, len(self.lambdas))
        ax.set_ylim(1, len(self.particle_snapshots[0]))
        ax.set_xlabel('iteration $i$')
        ax.set_ylabel('ESS')

    def plot_running_weights_traj(self, t=1, ax=None):
        ax = ax_setup(ax)
        ax.set_title('running log weights')
        ax.plot(self.current_log_weights[:t], alpha=0.05)
        ax.set_xlim(0, len(self.current_log_weights))
        ax.set_ylim(np.min(self.current_log_weights), np.max(self.current_log_weights))
        ax.set_xlabel('iteration $i$')
        ax.set_ylabel('running log importance weight')

    def plot_hist(self, t=1, ax=None):
        ax = ax_setup(ax)
        ax.set_title('log importance weight distribution')
        ax.hist(self.current_log_weights[t])

        ax.set_xlabel('log importance weight')
        ax.set_ylabel('counts')

    def plot_dashboard(self, t=100):
        plt.figure(figsize=(12, 8))

        scatter_ax = plt.subplot2grid((2,3), (0,0))
        self.scatter(t, global_ax_lock=False, ax=scatter_ax)
        scatter_ax.set_title('importance-weighted parameter samples\n(zoomed-in)')

        global_scatter_ax = plt.subplot2grid((2, 3), (1, 0))
        self.scatter(t, global_ax_lock=True, ax=global_scatter_ax)
        global_scatter_ax.set_title('importance-weighted parameter samples\n(big-picture)')

        lambdas_ax = plt.subplot2grid((2,3), (0,1))
        self.plot_lambdas(t, lambdas_ax)

        work_ax = plt.subplot2grid((2,3), (1,1))
        self.plot_running_weights_traj(t, work_ax)

        hist_ax = plt.subplot2grid((2,3), (0,2))
        self.plot_hist(t, hist_ax)

        ess_ax = plt.subplot2grid((2,3), (1,2))
        self.plot_ESS_traj(t, ess_ax)





if __name__ == '__main__':
    path = 'cess_smc_tinker_single_type_thresh=0.99,n_mcmc_steps=1,ll=gaussian,n_conf=2,resample_thresh=0.5,dataset=mini.npz'
    plotter = Plotter(path)
    plotter.plot_dashboard(-1)
    plt.show()


    # panels:
    # (1) scatter plot, colored by running weight
    # (2) lambdas
    # (3) ESS
    # (4) ESS increments
    # (5) work (traces, mean, stdev)
    # (6) histogram of running log weights


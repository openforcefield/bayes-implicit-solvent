from pickle import load
from bayes_implicit_solvent.continuous_parameter_experiments.gd_vs_langevin.autograd_based_experiment import n_types, Experiment, experiments, train_test_split, train_test_rmse, unreduce, expt_means, mbondi_model

types = mbondi_model.ordered_nodes

def load_expt_result(path):
    with open(path, 'rb') as f:
        result = load(f)
    return result


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


if __name__ == '__main__':



    from glob import glob
    ll = 'gaussian'
    fnames = glob('july_26_results/ll={}*'.format(ll))



    # TODO: n_types subplots
    # TODO: color by starting point and CV-fold
    def get_trajs(ll='gaussian'):
        fnames = glob('july_27_28_results/ll={}*'.format(ll))
        trajs = []
        starting_points = []
        alphas = []
        for i, fname in enumerate(fnames):
            result = load_expt_result(fname)
            starting_points.append(tuple(result['experiment'].x0))
            alphas.append(0.2 + 0.8 * (result['experiment'].cv_fold / 10))

            trajs.append(result['traj'])


    def plot_trajs(ax, type_ind=0, ll='gaussian'):
        fnames = glob('july_27_28_results/ll={}*'.format(ll))
        trajs = []
        starting_points = []
        alphas = []
        for i, fname in enumerate(fnames):
            result = load_expt_result(fname)
            starting_points.append(tuple(result['experiment'].x0))
            alphas.append(0.2 + 0.8 * (result['experiment'].cv_fold / 10))

            trajs.append(result['traj'])
        starting_point_set = list(set(starting_points))
        import seaborn as sns
        colors = sns.color_palette("hls", len(starting_point_set))
        color_dict = dict(zip(starting_point_set, colors))
        traj_colors = [color_dict[s] for s in starting_points]

        #X = np.vstack(trajs)
        #pca = PCA(2)
        #pca.fit(X)
        #for traj in trajs:
        #    x,y = pca.transform(traj).T
        #    plt.plot(x,y)

        for traj, traj_color, alpha in zip(trajs, traj_colors, alphas):
            x_nm,y = np.array(traj)[:,type_ind], np.array(traj)[:,n_types + type_ind]
            x = x_nm * 10
            ax.plot(x,y,c=traj_color, alpha=alpha)
            ax.scatter([x[0]], [y[0]], c=np.array(traj_color).reshape((1,3)))
            ax.set_xlabel('radius (Å)')
            ax.set_ylabel('scale')

    #ax = plt.subplot(3,3,1)
    from bayes_implicit_solvent.utils import remove_top_right_spines
    for ll in ['gaussian', 'student_t']:

        plt.figure(figsize=(12,12))
        ax = None
        for i in range(0, n_types):
            ax = plt.subplot(3,3,i+1, sharex=ax, sharey=ax)
            plot_trajs(ax, i, ll=ll)
            ax.set_title(types[i])
            remove_top_right_spines(ax)

        plt.tight_layout()
        plt.savefig('trajs_{}_traj_length=5000.png'.format(ll), bbox_inches='tight', dpi=300)
        plt.close()
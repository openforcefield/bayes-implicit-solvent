from pickle import load
from bayes_implicit_solvent.continuous_parameter_experiments.gd_vs_langevin.autograd_based_experiment import Experiment, experiments, train_test_split, train_test_rmse, unreduce, expt_means

def load_expt_result(path):
    with open(path, 'rb') as f:
        result = load(f)
    return result


import matplotlib.pyplot as plt
import numpy as np

expt_dataset = np.load('expt_dataset.npz')
expt_means = expt_dataset['expt_means']

def rmse(predictions, inds):
    pred_kcal_mol = unreduce(predictions[inds])
    expt_kcal_mol = unreduce(expt_means[inds])

    rmse = np.sqrt(np.mean((pred_kcal_mol - expt_kcal_mol) ** 2))
    return rmse


def train_test_rmse(predictions, split=0):
    train_inds, test_inds = train_test_split(split)
    train_rmse = rmse(predictions, train_inds)
    test_rmse = rmse(predictions, test_inds)
    return train_rmse, test_rmse

def compute_train_test_rmse_traj(prediction_traj, cv_fold=0):
    train_rmses = np.zeros(len(prediction_traj))
    test_rmses = np.zeros(len(prediction_traj))

    print(np.array(prediction_traj).shape)

    for i in range(len(prediction_traj)):
        train_rmses[i], test_rmses[i] = train_test_rmse(prediction_traj[i], cv_fold)
    return train_rmses, test_rmses


if __name__ == '__main__':

    plt.figure(figsize=(8,4))


    train_color = 'blue'
    test_color = 'green'
    result = load_expt_result('ll=student_t,k=1,hash(theta0)=9203586750394740867.pkl')

    train_rmses, test_rmses = compute_train_test_rmse_traj(result['prediction_traj'], result['experiment'].cv_fold)
    train_label = 'train'
    test_label = 'test'

    plt.plot(train_rmses, label=train_label, c=train_color, alpha=0.5)
    plt.plot(test_rmses, label=test_label, c=test_color, linestyle='--', alpha=0.5)


    plt.xlabel('iteration')
    plt.ylabel('RMSE (kcal/mol)')

    plt.savefig('train_test_rmses_0.png', dpi=300)
    plt.close()


    from glob import glob
    ll = 'gaussian'
    fnames = glob('july_26_results/ll={}*'.format(ll))
    def plot_scatter(path):
        result = load_expt_result(path)
        initial_pred = result['prediction_traj'][0]
        pred_kcalmol = unreduce(initial_pred)
        expt_kcalmol = unreduce(expt_means)
        plt.figure()
        plt.scatter(pred_kcalmol, expt_kcalmol)
        plt.savefig('scatter.png', dpi=300)
        plt.close()

    plot_scatter(fnames[0])


    def plot_result(ax, ll='gaussian'):
        fnames = glob('july_27_28_results/ll={}*'.format(ll))
        for i, fname in enumerate(fnames):
            result = load_expt_result(fname)
            train_rmses, test_rmses = compute_train_test_rmse_traj(result['prediction_traj'], result['experiment'].cv_fold)

            if i == 0:
                train_label = 'train'
                test_label = 'test'
            else:
                train_label = None
                test_label = None

            ax.plot(train_rmses, label=train_label, c=train_color, alpha=0.5)
            ax.plot(test_rmses, label=test_label, c=test_color, linestyle='--', alpha=0.5)


        ax.set_xlabel('iteration')
        ax.set_ylabel('RMSE (kcal/mol)')
        plt.legend(title='10-fold CV')

    ax = plt.subplot(1,2,1)
    plot_result(ax, 'gaussian')
    plt.title('gaussian likelihood')

    ax1 = plt.subplot(1,2,2, sharey=ax)
    plot_result(ax1, 'student_t')
    plt.title('student-t likelihood')

    plt.tight_layout()
    plt.savefig('train_test_rmses.png', dpi=300, bbox_inches='tight')
    plt.close()
from bayes_implicit_solvent.continuous_parameter_experiments.elemental_types_mh import *
from functools import partial
if __name__ == '__main__':
    from multiprocessing import Pool

    n_processes = 4
    pool = Pool(n_processes)


    def parallel_log_prob(theta):
        return sum(pool.map(partial(log_prob_component, theta=theta), range(len(vacuum_trajs))))




    theta0 = pack(initial_radii, initial_scales)
    print('initial theta', theta0)
    initial_log_prob = parallel_log_prob(theta0)
    print('initial log prob', log_prob(theta0))

    stepsize = 0.5 * 1e-1
    n_steps = 10000
    dim_to_perturb = 5

    traj, log_probs, acceptance_fraction = sparse_mh(theta0, parallel_log_prob, n_steps=n_steps, stepsize=stepsize,
                                                     dim_to_perturb=dim_to_perturb)

    np.savez(os.path.join(data_path,
                          'elemental_types_mh_freesolv_{}.npz'.format(
                              name)),
             traj=traj, log_probs=log_probs, acceptance_fraction=acceptance_fraction, stepsize=stepsize,
             n_steps=n_steps, dim_to_perturb=dim_to_perturb)
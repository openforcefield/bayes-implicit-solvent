from jax_posterior import log_posterior, grad_log_posterior, sample_prior, get_predictions
from jax import numpy as np
import numpy as onp
from tqdm import tqdm
from bayes_implicit_solvent.samplers import random_walk_mh

def gradient_descent(jac, x0, step_size, num_iters, callback=None):
    """

    Parameters
    ----------
    jac : callable, with signature jac(x)
    x0 : array
    step_size : float
    n_steps : int
    callback: callable, with signature callback(x, i, g)

    Returns
    -------

    """
    traj = [x0]
    for i in tqdm(range(num_iters)):
        g = jac(traj[-1])
        traj.append(traj[-1] - step_size * g)
        if np.isnan(traj[-1]).any():
            print('NaN encountered!')
            return traj
        if not (callback is None):
            callback(traj[-1], i, g)
    return traj

if __name__ == '__main__':
    perturbation_sigma = 0.2
    pred_traj_thinning = 50
    n_steps = 10000
    #n_steps = 100
    step_size =1e-5

    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No valid job_id supplied! Selecting one at random")
        job_id = onp.random.randint(10000)

    onp.random.seed(job_id)

    obc2_theta = np.array([
        1.5, 1.2, 1.7, 1.55, 1.5, 1.5, 2.1, 1.85, 1.8,
        0.8, 0.85, 0.72, 0.79, 0.85, 0.88, 0.8, 0.86, 0.96])

    x0 = obc2_theta + onp.random.randn(len(obc2_theta)) * perturbation_sigma

    prior_sample = sample_prior()


    jac = lambda x : - grad_log_posterior(x)
    gd_traj = gradient_descent(jac, x0, step_size=step_size, num_iters=n_steps)

    prediction_traj = onp.array(list(map(get_predictions, gd_traj[::pred_traj_thinning])))

    onp.savez('gd_starting_from_obc2_perturbed_by_sigma={},job_id={}'.format(perturbation_sigma, job_id),
              random_seed=job_id,
              rw_mh_traj=onp.asarray(gd_traj),
              prediction_traj=prediction_traj)

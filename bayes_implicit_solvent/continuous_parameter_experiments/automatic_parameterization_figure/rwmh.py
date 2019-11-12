from jax_posterior import log_posterior, grad_log_posterior, sample_prior, get_predictions
from jax import numpy as np
import numpy as onp
from bayes_implicit_solvent.samplers import random_walk_mh

if __name__ == '__main__':
    perturbation_sigma = 0.2
    pred_traj_thinning = 50
    #n_steps = 10000
    n_steps = 100
    step_size = 0.005

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

    rw_mh_traj, rw_mh_post_traj, accept_rate = random_walk_mh(x0, log_posterior, n_steps=n_steps, stepsize=step_size)

    prediction_traj = onp.array(list(map(get_predictions, rw_mh_traj[::pred_traj_thinning])))

    onp.savez('rw_mh_starting_from_obc2_perturbed_by_sigma={},job_id={}'.format(perturbation_sigma, job_id),
              random_seed=job_id,
              rw_mh_traj=onp.asarray(rw_mh_traj),
              rw_mh_post_traj=onp.asarray(rw_mh_post_traj),
              prediction_traj=prediction_traj)

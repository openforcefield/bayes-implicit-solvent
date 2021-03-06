from bayes_implicit_solvent.marginal_likelihood.two_types_forward_ais import \
    annealed_log_posterior_at_multiple_values_of_beta, annealed_log_posterior


from numpy import load, random, savez
from jax import numpy as np

posterior_sample_result = load('two_types_posterior_samples.npz')
posterior_samples = posterior_sample_result['traj']


def sample_from_posterior():
    return posterior_samples[random.randint(len(posterior_samples))]

#from numpy import load
#optimized_reverse_betas = load('../../notebooks/optimized_reverse_betas.npz')['optimized_reverse_betas_1000']

from bayes_implicit_solvent.samplers import random_walk_mh
from tqdm import tqdm

if __name__ == "__main__":
    N_trajectories = 1000
    N_annealing_steps = 100

    trajectories = []
    log_weight_trajs = []

    for k in range(N_trajectories):

        theta = sample_from_posterior()

        traj = [theta]
        log_weights = [0]

        betas = np.linspace(1, 0, N_annealing_steps)
        #betas = optimized_betas
        #betas = optimized_reverse_betas

        trange = tqdm(range(1, N_annealing_steps))
        for t in trange:
            log_pdf_tminus1, log_pdf_t = annealed_log_posterior_at_multiple_values_of_beta(traj[-1],
                                                                                                   [betas[t - 1],
                                                                                                    betas[t]])
            log_weights.append(log_weights[t - 1] + (log_pdf_t - log_pdf_tminus1))

            log_prob_fun = lambda theta: annealed_log_posterior(theta, betas[t])
            mh_traj, _, acceptance_prob = random_walk_mh(traj[-1], log_prob_fun, n_steps=50, stepsize=0.015, progress_bar=False)

            traj.append(mh_traj[-1])
            trange.set_postfix(running_log_Z_estimate=log_weights[-1], acceptance_prob=acceptance_prob)

        trajectories.append(np.array(traj))
        log_weight_trajs.append(np.array(log_weights))

        savez('two_types_reverse_ais.npz',
                 trajectories=trajectories,
                 log_weight_trajectories=log_weight_trajs)

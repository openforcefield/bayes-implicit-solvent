from bayes_implicit_solvent.marginal_likelihood.two_types_forward_ais import rejection_sample_from_prior, log_prior, log_likelihood, annealed_log_posterior,\
    ll, n_conf, dataset
import numpy as np
from bayes_implicit_solvent.marginal_likelihood.single_type_forward_smc import CESS_SMC

if __name__ == '__main__':
    np.random.seed(0)
    n_particles = 100
    print("ll, n_conf, dataset, n_particles", ll, n_conf, dataset, n_particles)

    initial_particles = np.array([rejection_sample_from_prior() for _ in range(n_particles)])

    thresh = 0.99
    n_mcmc_steps = 5
    resample_thresh = 0.5
    lambdas, particle_snapshots, log_q_s, incremental_log_weights, current_log_weights, stepsizes, acceptance_rates = \
        CESS_SMC(initial_particles,
                 log_prior=log_prior, log_likelihood=log_likelihood, annealed_log_posterior=annealed_log_posterior,
                 thresh=thresh, resample_thresh=resample_thresh, n_mcmc_steps=n_mcmc_steps)
    np.savez('cess_smc_tinker_two_types_thresh={},n_mcmc_steps={},ll={},n_conf={},resample_thresh={},dataset={}.npz'.format(thresh, n_mcmc_steps, ll, n_conf, resample_thresh, dataset),
             lambdas=lambdas,
             particle_snapshots=particle_snapshots,
             log_q_s=np.array(log_q_s),
             incremental_log_weights=np.array(incremental_log_weights),
             current_log_weights=np.array(current_log_weights),
             stepsizes=stepsizes,
             acceptance_rates=acceptance_rates,
             )

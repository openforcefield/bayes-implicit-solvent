import numpy as np

from bayes_implicit_solvent.marginal_likelihood.two_types_radius_and_scale_forward_ais import prior_location, log_posterior
from bayes_implicit_solvent.samplers import random_walk_mh

if __name__ == '__main__':
    traj, log_probs, acceptance_probability = random_walk_mh(prior_location, log_prob_fun=log_posterior, n_steps=100000,
                                                             stepsize=0.015,
                                                             progress_bar=True)
    np.savez('two_types_radius_and_scale_posterior_samples.npz',
             traj=traj,
             log_probs=log_probs,
             acceptance_probability=acceptance_probability,
             )

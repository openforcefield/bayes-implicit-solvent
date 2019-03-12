from jax import jit, vmap
from jax import numpy as np
from numpy import random as npr
from scipy.stats import t as student_t, norm

from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized

from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp

dataset = "mini"

if dataset == "mini":
    from bayes_implicit_solvent.marginal_likelihood.mini_freesolv_ch import mols
elif dataset == "tiny":
    from bayes_implicit_solvent.marginal_likelihood.tiny_freesolv_ch import mols
else:
    raise(RuntimeError)


@jit
def predict_solvation_free_energy_jax(theta, distance_matrices, charges):
    radius, scale = theta
    N = len(charges)

    @jit
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radius * np.ones(N), scale * np.ones(N), charges)

    W_F = vmap(compute_component)(distance_matrices)

    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)

@jit
def get_predictions(theta):
    return np.array(
        [predict_solvation_free_energy_jax(theta, mol.distance_matrices, mol.charges) for (i, mol) in
         enumerate(mols)])


expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncs = np.array([mol.experimental_uncertainty for mol in mols])

prior_location = np.array([0.15, 0.8])
radius_lower_bound = 0.01
scale_lower_bound = 0.1


def in_bounds(theta):
    radius, scale = theta
    return (radius >= radius_lower_bound) and (scale >= scale_lower_bound)


def log_prior(theta):
    """
    TODO: double-check if it's okay to leave this unnormalized...
    """
    if not in_bounds(theta):
        return - np.inf
    return np.sum(norm.logpdf(theta, loc=prior_location))


def log_likelihood_of_predictions(predictions):
    return np.sum(student_t.logpdf(predictions, loc=expt_means,
                                   scale=expt_uncs,
                                   df=7))


def log_likelihood(theta):
    return log_likelihood_of_predictions(get_predictions(theta))


def rejection_sample_from_prior():
    theta = npr.randn(2) + prior_location
    while not in_bounds(theta):
        theta = npr.randn(2) + prior_location
    return theta


def annealed_log_posterior_at_multiple_values_of_beta(theta, betas=[0.0, 1.0]):
    log_prior_ = log_prior(theta)

    log_likelihood_ = log_likelihood(theta)

    log_posterior_ = log_prior_ + log_likelihood_

    return [((1 - beta) * log_prior_) + (beta * log_posterior_) for beta in betas]


def log_posterior(theta):
    return log_prior(theta) + log_likelihood(theta)


def annealed_log_posterior(theta, beta=1.0):
    """beta=0 --> log_prior, beta=1 --> log_posterior"""
    return annealed_log_posterior_at_multiple_values_of_beta(theta, [beta])[0]


from bayes_implicit_solvent.samplers import random_walk_mh
from tqdm import tqdm

#from numpy import load
#optimized_betas = load('../../notebooks/optimized_betas.npz')['optimized_betas_1000']

if __name__ == "__main__":
    N_trajectories = 1000
    N_annealing_steps = 100

    trajectories = []
    log_weight_trajs = []

    for k in range(N_trajectories):

        theta = rejection_sample_from_prior()

        traj = [theta]
        log_weights = [0]

        betas = np.linspace(0, 1, N_annealing_steps)
        #betas = optimized_betas

        trange = tqdm(range(1, N_annealing_steps))
        for t in trange:
            log_pdf_tminus1, log_pdf_t = annealed_log_posterior_at_multiple_values_of_beta(traj[-1],
                                                                                           [betas[t - 1], betas[t]])
            log_weights.append(log_weights[t - 1] + (log_pdf_t - log_pdf_tminus1))

            log_prob_fun = lambda theta: annealed_log_posterior(theta, betas[t])
            mh_traj, _, _ = random_walk_mh(traj[-1], log_prob_fun, n_steps=50, stepsize=0.015, progress_bar=False)

            traj.append(mh_traj[-1])

            trange.set_postfix(running_log_Z_estimate=log_weights[-1])

        trajectories.append(np.array(traj))
        log_weight_trajs.append(np.array(log_weights))

        import numpy as np

        np.savez('single_type_forward_ais_optimized_protocol_longer_{}.npz'.format(dataset),
                 trajectories=trajectories,
                 log_weight_trajectories=log_weight_trajs,
                 notes="""assumes incorrectly that initial distribution is normalized!"""
                 )

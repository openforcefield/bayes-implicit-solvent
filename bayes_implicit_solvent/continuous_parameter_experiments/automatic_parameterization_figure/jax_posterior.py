# imports
from pickle import load

import jax.numpy as np
from jax import grad, jit
from jax.scipy.stats import norm

import numpy as onp
from tqdm import tqdm

# read input pickle archive
# from https://github.com/openforcefield/bayes-implicit-solvent/raw/fb54fda2ac1a079143b59476153b2be33214244e/bayes_implicit_solvent/continuous_parameter_experiments/gd_vs_langevin/freesolv_inputs.pkl
with open('freesolv_inputs.pkl', 'rb') as f:
  freesolv_inputs = load(f)
print(list(freesolv_inputs.keys()))

# load from dictionary
distance_matrices = freesolv_inputs['distance_matrices'] # list of (25,n_atoms_i,n_atoms_i) arrays
print('distance_matrices[0].shape', distance_matrices[0].shape)
type_slices = freesolv_inputs['type_slices']
charges = freesolv_inputs['charges']
n_types = freesolv_inputs['n_types']

# thinning
mol_thinning = 1
snapshot_thinning = 1

distance_matrices = [d[::snapshot_thinning] for d in tqdm(distance_matrices[::mol_thinning])]
type_slices = [t for t in tqdm(type_slices[::mol_thinning])]
charges = [c for c in tqdm(charges[::mol_thinning])]
expt_means = freesolv_inputs['expt_means'][::mol_thinning]
expt_uncertainties = freesolv_inputs['expt_uncertainties'][::mol_thinning]

print('distance_matrices[0].shape', distance_matrices[0].shape)

# define energy function
@jit
def step(x):
    # return (x > 0)
    return 1.0 * (x >= 0)

@jit
def compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges,
                                  offset=0.009, screening=138.935484, surface_tension=28.3919551,
                                  solvent_dielectric=78.5, solute_dielectric=1.0,
                                  ):
    """Compute GBSA-OBC energy from a distance matrix"""
    N = len(radii)
    #print(type(distance_matrix))
    eye = np.eye(N, dtype=distance_matrix.dtype)
    #print(type(eye))
    r = distance_matrix + eye # so I don't have divide-by-zero nonsense
    or1 = radii.reshape((N, 1)) - offset
    or2 = radii.reshape((1, N)) - offset
    sr2 = scales.reshape((1, N)) * or2

    L = np.maximum(or1, abs(r - sr2))
    U = r + sr2
    I = step(r + sr2 - or1) * 0.5 * (
            1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * np.log(
        L / U) / r)

    I -= np.diag(np.diag(I))
    I = np.sum(I, axis=1)

    # okay, next compute born radii
    offset_radius = radii - offset
    psi = I * offset_radius
    psi_coefficient = 0.8
    psi2_coefficient = 0
    psi3_coefficient = 2.909125

    psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - np.tanh(psi_term) / radii)

    # finally, compute the three energy terms
    E = 0.0

    # single particle
    E += np.sum(surface_tension * (radii + 0.14) ** 2 * (radii / B) ** 6)
    E += np.sum(-0.5 * screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B)

    # particle pair
    f = np.sqrt(r ** 2 + np.outer(B, B) * np.exp(-r ** 2 / (4 * np.outer(B, B))))
    charge_products = np.outer(charges, charges)

    E += np.sum(np.triu(-screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f, k=1))

    return E

# call once on each molecule, since each n_atoms is separately JIT'd...

for i in tqdm(range(len(distance_matrices))):
    distance_matrix = distance_matrices[i][0]
    n = len(distance_matrix)
    compute_OBC_energy_vectorized(distance_matrix, np.ones(n), np.ones(n), np.ones(n))

# Free energy as a function of parameters"""

kj_mol_to_kT = 0.40339497740718383

from jax import vmap
from jax.scipy.special import logsumexp

def one_sided_exp(w_F):
  return - (logsumexp(- w_F) - np.log(len(w_F)))

theta = np.hstack((np.ones(n_types), np.ones(n_types)))

def unpack(theta):
    radii_, scaling_factors_ = theta[:n_types], theta[n_types:]
    return 0.1 * radii_, scaling_factors_

@jit
def predict_solvation_free_energy_jax(theta, distance_matrices, charges, type_slice):
    radii_, scaling_factors_ = unpack(theta)

    radii, scaling_factors = radii_[type_slice], scaling_factors_[type_slice]

    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)
    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)


#@jit
def get_predictions(theta):
    """Produce predictions for all molecules in the freesolv set"""
    return np.array([predict_solvation_free_energy_jax(theta, distance_matrices[i], charges[i], type_slices[i]) for i in range(len(distance_matrices))])
# TODO: Replace with vmap?

kj_mol_to_kcal_mol = 0.2390057361376673
kT_to_kcal_mol = (1.0 / kj_mol_to_kT) * kj_mol_to_kcal_mol

# define prior, likelihood, and posterior

r_bounds = (0.25, 3) # angstroms
s_bounds = (0.01, 2)
def uniform_log_prior(theta):
    radii_in_nm, scales = unpack(theta)
    radii = radii_in_nm * 10
    n = len(radii)
    r_in_bounds = (min(radii) >= r_bounds[0]) and (max(radii) <= r_bounds[1])
    s_in_bounds = (min(scales) >= s_bounds[0]) and (max(scales) <= s_bounds[1])
    if r_in_bounds and s_in_bounds:
        return - np.log(n*(r_bounds[1] - r_bounds[0])) - np.log(n * (s_bounds[1] - s_bounds[0]))
    else:
        return - np.inf

def sample_prior():
    r = r_bounds[0] + onp.random.rand(n_types) * (r_bounds[1] - r_bounds[0])
    s = s_bounds[0] + onp.random.rand(n_types) * (s_bounds[1] - s_bounds[0])
    return np.hstack([r, s])

# likelihood
#@jit
from jax.scipy.stats import t as student_t
def log_likelihood_as_fxn_of_prediction(prediction, expt_mean, expt_uncertainty):
    return student_t.logpdf(prediction, loc=expt_mean, df=7)
    #return norm.logpdf(prediction, loc=expt_mean, scale=expt_uncertainty)

#@jit
def compute_log_likelihood_component(theta, distance_matrix_array, charge_array, type_slice_array, expt_mean, expt_uncertainty):
    prediction = predict_solvation_free_energy_jax(theta, distance_matrix_array, charge_array, type_slice_array)
    return log_likelihood_as_fxn_of_prediction(prediction, expt_mean, expt_uncertainty)

print('calling compute_log_likelihood_component once per component')
for i in tqdm(range(len(distance_matrices))):
    compute_log_likelihood_component(theta, distance_matrices[i], charges[i], type_slices[i], expt_means[i], expt_uncertainties[i])

g_component = jit(grad(compute_log_likelihood_component))

all_inds = np.arange(len(distance_matrices))

#@jit
def compute_grad_log_likelihood_w_loop(theta):
    g = np.zeros(len(theta))
    for i in range(len(distance_matrices)):
        g += g_component(theta, distance_matrices[i], charges[i], type_slices[i], expt_means[i], expt_uncertainties[i])
    return g

from jax.lax import scan
#@jit
def compute_grad_log_likelihood_w_vmap(theta):
    carry = np.zeros(len(theta))
    xs = [(distance_matrices[i], charges[i], type_slices[i], expt_means[i], expt_uncertainties[i]) for i in range(len(distance_matrices))]
    def f(x):
        g = g_component(theta, *x)
        return g
    return np.sum(vmap(f)(xs), 0)

def compute_grad_log_likelihood_w_scan(theta):
    carry = np.zeros(len(theta))
    xs = [(distance_matrices[i], charges[i], type_slices[i], expt_means[i], expt_uncertainties[i]) for i in range(len(distance_matrices))]
    def f(carry, x):
        g = g_component(theta, *x)
        return carry + g, g
    return scan(f, carry, xs)

#@jit # do NOT jit this! will consume all memory
def grad_log_likelihood(theta):
    #return compute_grad_log_likelihood_w_vmap(theta)
    #return compute_grad_log_likelihood_w_scan(theta)
    return compute_grad_log_likelihood_w_loop(theta)


# posterior
def log_posterior(theta):
    log_prior_val = uniform_log_prior(theta)
    if np.isfinite(log_prior_val):
        preds = get_predictions(theta)
        log_lik_val = np.sum(log_likelihood_as_fxn_of_prediction(preds, expt_means, expt_uncertainties))
        return log_prior_val + log_lik_val
    else:
        return - np.inf

#@jit # do NOT jit this!
def grad_log_posterior(theta):
    log_prior_val = uniform_log_prior(theta)
    if np.isfinite(log_prior_val):
        return grad_log_likelihood(theta)
    else:
        return np.nan * theta
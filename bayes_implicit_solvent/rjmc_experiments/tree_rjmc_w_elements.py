from jax.config import config
config.update("jax_enable_x64", True)

from numpy import load, random
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule

import sys

valid_lls = ['student-t']

try:
    job_id = int(sys.argv[1])
    ll = sys.argv[2]
    assert(ll in valid_lls)
except:
    print("Didn't parse input, selecting job parameters at random")
    job_id = random.randint(10000)
    ll = valid_lls[random.randint(len(valid_lls))]


def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj


from glob import glob
from pkg_resources import resource_filename

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)
from numpy import random
#random.seed(job_id)
#random.shuffle(paths_to_samples)
# 90:10 split

# TODO: Consider running on a 90:10 train/test split again...
paths_to_samples = paths_to_samples#[:int(0.9*len(paths_to_samples))]

print('number of molecules being considered: {}'.format(len(paths_to_samples)))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))
print('first few CIDs', cids[:5])

mols = []

# TODO: potentially adjust this -- using n_configuration_samples=25 for faster debugging...
n_configuration_samples = 25

from bayes_implicit_solvent.freesolv import cid_to_smiles

from bayes_implicit_solvent.constants import beta
def unreduce(value):
    """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
    return value / (beta * unit.kilocalorie_per_mole)

for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning])

    if (unreduce(mol.experimental_value) > -15) and (unreduce(mol.experimental_value) < 5):
        mols.append(mol)
    else:
        print('discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles, cid))

oemols = [mol.mol for mol in mols]
from jax import numpy as np

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp



def unpack(theta):
    n_types = int(len(theta) / 2)
    radii_, scaling_factors_ = theta[:n_types], theta[n_types:]
    return 0.1 * radii_, scaling_factors_


@jit
def predict_solvation_free_energy_jax(radii, scaling_factors, distance_matrices, charges):
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)
    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)

distance_matrices = [mol.distance_matrices for mol in mols]
charges = [mol.charges for mol in mols]

expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncertainties = np.array([mol.experimental_uncertainty for mol in mols])

from bayes_implicit_solvent.typers import RADIUS_UNIT

from bayes_implicit_solvent.typers import AtomSpecificationProposal

from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

initial_tree = mbondi_model
initial_tree.remove_node('[#14]') # otherwise everything is -inf, because this type will be empty
initial_tree.proposal_sigmas['radius'] = 1e-2 * RADIUS_UNIT
initial_tree.proposal_sigmas['scale_factor'] = 1e-2

# add one more parameter per element appearing in FreeSolv but not specified in obc2 parameter set to initial tree
for i in [17, 35, 53]:
    smirks = '[#{}]'.format(i)
    initial_tree.add_child(smirks, '*')
    initial_tree.un_delete_able_types.add(smirks)

ring_specifiers = ['r0', 'r3', 'r4', 'r5', 'r6', 'r7', 'a', 'A']
charge_specifiers = ['-1', '+0', '+1', '+2']
hydrogen_count_specifiers = ['H0', 'H1', 'H2', 'H3', 'H4']
connectivity_specifiers = ['X1', 'X2', 'X3', 'X4']


specifiers = ring_specifiers + charge_specifiers + hydrogen_count_specifiers + connectivity_specifiers
atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
smirks_elaboration_proposal = atom_specification_proposal

print('initial tree:')
print(initial_tree)

name = 'tree_rjmc_n_config={}_{}_ll'.format(n_configuration_samples, ll)

from bayes_implicit_solvent.prior_checking import check_no_empty_types

error_y_trees = []

# handle the RJ moves differently from the random-walk moves

from jax.scipy.stats import norm

radius_prior = 1.5
scale_prior = 0.8

@jit
def log_prior(theta):
    radii, scales = unpack(theta)
    return np.sum(norm.logpdf(radii - radius_prior)) + np.sum(norm.logpdf(scales - scale_prior))

# todo: save predictions on all train and test datapoints?
def get_predictions(theta, types):
    radii_, scaling_factors_ = unpack(theta)

    radii = [radii_[types[i]] for i in range(len(types))]
    scaling_factors = [scaling_factors_[types[i]] for i in range(len(types))]
    return np.array([predict_solvation_free_energy_jax(radii[i], scaling_factors[i], distance_matrices[i], charges[i]) for i in range(len(charges))])


def get_radii_scales(theta, type_slices):
    radii_, scaling_factors_ = unpack(theta)
    radii = [radii_[type_slice] for type_slice in type_slices]
    scaling_factors = [scaling_factors_[type_slice] for type_slice in type_slices]
    return radii, scaling_factors

if ll == 'student-t':
    def log_likelihood_as_fxn_of_prediction(prediction, expt_mean, expt_uncertainty):
        return student_t.logpdf(prediction, loc=expt_mean, scale=expt_uncertainty, df=7)
elif ll == 'gaussian':
    def log_likelihood_as_fxn_of_prediction(prediction, expt_mean, expt_uncertainty):
        return norm.logpdf(prediction, loc=expt_mean, scale=expt_uncertainty)

def compute_log_likelihood_component(theta, type_slice_array, distance_matrix_array, charge_array, expt_mean, expt_uncertainty):
    r, s = unpack(theta)
    radius_array, scale_array = r[type_slice_array], s[type_slice_array] # TODO: replace with index_update...
    prediction = predict_solvation_free_energy_jax(radius_array, scale_array, distance_matrix_array, charge_array)
    return log_likelihood_as_fxn_of_prediction(prediction, expt_mean, expt_uncertainty)

from jax import jit, grad

g_component = jit(grad(compute_log_likelihood_component))

all_inds = np.arange(len(distance_matrices))

def compute_grad_log_likelihood_w_loop(theta, type_slices):
    g = np.zeros(len(theta))
    for i in range(len(distance_matrices)):
        g += g_component(theta, type_slices[i], distance_matrices[i], charges[i], expt_means[i], expt_uncertainties[i])
    return g

from jax.lax import fori_loop


# TODO: complete and jit this...
def compute_grad_log_likelihood_w_fori_loop(theta, type_slices):
    g = np.zeros(len(theta))

    def g_component_update(i, g):
        return g + g_component(theta, type_slices[i], distance_matrices[i], charges[i], expt_means[i],
                               expt_uncertainties[i])
    g = fori_loop(0, len(distance_matrices), g_component_update, g)
    return g

# TODO: gradcheck
def grad_log_likelihood(theta, type_slices):
    return compute_grad_log_likelihood_w_loop(theta, type_slices)

# posterior
def log_posterior(theta, type_slices):
    log_prior_val = log_prior(theta)
    if np.isfinite(log_prior_val):
        preds = get_predictions(theta, type_slices)
        log_lik_val = np.sum(log_likelihood_as_fxn_of_prediction(preds, expt_means, expt_uncertainties))
        return log_prior_val + log_lik_val
    else:
        return - np.inf

#@jit # do NOT jit this!
def grad_log_posterior(theta, type_slices):
    log_prior_val = log_prior(theta)
    if np.isfinite(log_prior_val):
        return grad(log_prior)(theta) + grad_log_likelihood(theta, type_slices)
    else:
        return np.nan * theta

def get_theta(tree):
    # TODO: don't be so gross about unit conversions...
    return np.hstack([10 * tree.get_radii(), tree.get_scale_factors()])


from jax.scipy.stats import t as student_t
def log_prob(tree):
    log_prior_value = check_no_empty_types(tree)

    theta = get_theta(tree)

    log_prior_value += log_prior(theta)

    if log_prior_value > -np.inf:
        types = tree.apply_to_molecule_list(oemols)
        predictions = get_predictions(theta, types)

        log_likelihood_value = np.sum(log_likelihood_as_fxn_of_prediction(predictions, expt_means, expt_uncertainties))
        return log_prior_value + log_likelihood_value
    else:
        return log_prior_value



from bayes_implicit_solvent.samplers import tree_rjmc

n_within_model_steps_per_cross_model_proposal = 10
n_cross_model_proposals = 100
n_iterations = n_within_model_steps_per_cross_model_proposal * n_cross_model_proposals

trajs = []

from tqdm import tqdm

trange = tqdm(range(n_cross_model_proposals))

def make_one_rjmc_proposal(tree):
    result = tree_rjmc(tree, log_prob, smirks_elaboration_proposal, n_iterations=1,
                       fraction_cross_model_proposals=1.0, progress_bar=False)
    return result['traj'][-1]


tree_traj = [initial_tree]
n_types_traj = [initial_tree.number_of_nodes]
within_model_trajs = []
prediction_traj = []

from bayes_implicit_solvent.samplers import langevin

train_smiles = [mol.smiles for mol in mols]

def save():
    name = 'tree_rjmc_from_wildcard_n_iter={}_ll={}_job_id={}'.format(
        n_iterations,
        ll,
        job_id
    )
    onp.savez(name + '.npz',
             ll=ll,
             job_id=job_id,
             train_smiles=onp.array(train_smiles),
             n_types_traj=n_types_traj,
             within_model_trajs=within_model_trajs,
             expt_means=expt_means,
             expt_uncs=expt_uncertainties,
             )

    from pickle import dump
    with open(name + '.pkl', 'wb') as f:
        dump(tree_traj, f)

from copy import deepcopy
from jax import grad
import numpy as onp

kj_mol_to_kT = 0.40339497740718383
kj_mol_to_kcal_mol = 0.2390057361376673
kT_to_kcal_mol = (1.0 / kj_mol_to_kT) * kj_mol_to_kcal_mol

def get_rmse_in_kcal_per_mol(preds):
    expt_means_in_kcal_per_mol = expt_means * kT_to_kcal_mol
    preds_in_kcal_per_mol = preds * kT_to_kcal_mol

    return float(np.sqrt(np.mean((preds_in_kcal_per_mol - expt_means_in_kcal_per_mol) ** 2)))


for chunk in trange:
    tree = make_one_rjmc_proposal(tree_traj[-1])
    types = tree.apply_to_molecule_list(oemols)
    theta0 = get_theta(tree)
    # should just change to using consistent units throughout!!

    N = int(len(theta0) / 2)
    stepsize = 0.001
    if ll == 'gaussian':
        stepsize *= 0.25

    def within_model_log_prob(theta):
        return log_posterior(theta, types)

    def within_model_grad_log_prob(theta):
        return grad_log_posterior(theta, types)


    def run_langevin(theta0, stepsize=stepsize):
        v0 = onp.random.randn(*theta0.shape)
        within_model_traj = langevin(theta0, v0, within_model_log_prob, within_model_grad_log_prob,
                                     n_steps=n_within_model_steps_per_cross_model_proposal,
                                     stepsize=stepsize,
                                     collision_rate=0.001/stepsize)
        current_log_prob = within_model_log_prob(within_model_traj[-1])
        return within_model_traj, current_log_prob
    within_model_traj, current_log_prob = run_langevin(theta0, stepsize)
    while not np.isfinite(current_log_prob):
        print("that didn't go well! trying again with smaller stepsize...")
        print("\told stepsize: ", stepsize)
        stepsize *= 0.5
        print("\tnew stepsize ", stepsize)
        within_model_traj, current_log_prob = run_langevin(theta0, stepsize)

    theta = within_model_traj[-1]
    r, s = unpack(theta)
    tree.set_radii(r)
    tree.set_scale_factors(s)

    tree_traj.append(deepcopy(tree))


    predictions = get_predictions(within_model_traj[-1], types)
    prediction_traj.append(predictions)
    train_rmse = get_rmse_in_kcal_per_mol(predictions)

    trange.set_postfix(
        current_log_prob=current_log_prob,
        current_train_rmse=train_rmse,
        max_n_types=max(n_types_traj),
        min_n_types=min(n_types_traj),
    )

    n_types_traj.append(N)

    for t in within_model_traj:
        within_model_trajs.append(t)

    if (chunk + 1) % 100 == 0:
        save()

save()

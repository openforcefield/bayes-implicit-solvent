from bayes_implicit_solvent.molecule import Molecule
from simtk import unit
from numpy import load


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
random.seed(0)
random.shuffle(paths_to_samples)
paths_to_samples = paths_to_samples[::2]

print('number of molecules being considered: {}'.format(len(paths_to_samples)))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))
print('first few CIDs', cids[:5])

mols = []

n_configuration_samples = 15

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
import numpy as np
element_inds = []
all_elements = ['S', 'Cl', 'F', 'C', 'I', 'N', 'Br', 'H', 'P', 'O']
N = len(all_elements)
element_dict = dict(zip(all_elements, range(len(all_elements))))

initial_radius_dict = dict(H=0.12, C=0.17, N=0.155, O=0.15, F=0.15,
                   P=0.185, S=0.18, Cl=0.17, Br=0.15, I=0.15)
initial_scaling_factor_dict = dict(H=0.85, C=0.72, N=0.79, O=0.85, F=0.88,
                           P=0.86, S=0.96, Cl=0.80, Br=0.80, I=0.80)


for mol in mols:
    element_inds.append(np.array([element_dict[a.element.symbol] for a in list(mol.top.atoms())]))

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp

@jit
def predict_solvation_free_energy_jax(theta, distance_matrices, charges, element_ind_array):
    radii_, scaling_factors_ = theta[:N], theta[N:]

    radii = radii_[element_ind_array]
    scaling_factors = scaling_factors_[element_ind_array]

    @jit
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)

    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)

distance_matrices = [mol.distance_matrices for mol in mols]
charges = [mol.charges for mol in mols]

expt_means = unreduce(np.array([mol.experimental_value for mol in mols]))
expt_uncs = unreduce(np.array([mol.experimental_uncertainty for mol in mols]))

from bayes_implicit_solvent.typers import RADIUS_UNIT

from bayes_implicit_solvent.freesolv import smiles_list
from bayes_implicit_solvent.typers import AtomSpecificationProposal

np.random.seed(0)

from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

initial_tree = mbondi_model
initial_tree.remove_node('[#14]') # otherwise everything is -inf, because this type will be empty
initial_tree.proposal_sigmas['radius'] = 5 * 1e-2 * RADIUS_UNIT
initial_tree.proposal_sigmas['scale_factor'] = 5 * 1e-2

# add one more parameter per element appearing in FreeSolv but not specified in obc2 parameter set to initial tree
for i in [17, 35, 53]:
    smirks = '[#{}]'.format(i)
    initial_tree.add_child(smirks, '*')
    initial_tree.un_delete_able_types.add(smirks)

specifiers = ['X1', 'X2', 'X3', 'X4', 'a', 'A', '-1', '+0', '+1', '+2']
atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
smirks_elaboration_proposal = atom_specification_proposal

print('initial tree:')
print(initial_tree)

ll = 'student-t'

import os

name = 'tree_rjmc_n_config={}_{}_ll'.format(n_configuration_samples, ll)

from bayes_implicit_solvent.prior_checking import check_no_empty_types

error_y_trees = []

# handle the RJ moves differently from the random-walk moves

from jax.scipy.stats import norm

radius_prior = 0.15
scale_prior = 0.8

@jit
def log_prior(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]
    return np.sum(norm.logpdf(radii - radius_prior)) + np.sum(norm.logpdf(scales - scale_prior))


def get_predictions(theta, types):
    return np.array([predict_solvation_free_energy_jax(theta, distance_matrices[i], charges[i], types[i]) for i in range(len(charges))])


def log_likelihood_of_predictions(predictions):
    if ll == 'student-t':
        log_likelihood_value = np.sum(student_t.logpdf(predictions - expt_means, scale=expt_uncs, df=7))
    elif ll == 'gaussian':
        log_likelihood_value = np.sum(norm.logpdf(predictions - expt_means, scale=expt_uncs))
    else:
        raise (RuntimeError('invalid ll value'))
    return log_likelihood_value

from scipy.stats import t as student_t
def log_prob(tree):
    log_prior_value = check_no_empty_types(tree)

    theta = np.hstack([tree.get_radii(), tree.get_scale_factors()])

    log_prior_value += log_prior(theta)

    if log_prior_value > -np.inf:
        types = tree.apply_to_molecule_list(oemols)
        predictions = get_predictions(theta, types)

        log_likelihood_value = log_likelihood_of_predictions(predictions)
        return log_prior_value + log_likelihood_value
    else:
        return log_prior_value


from bayes_implicit_solvent.samplers import tree_rjmc

n_within_model_steps_per_cross_model_proposal = 100
n_cross_model_proposals = 1000
n_iterations = n_within_model_steps_per_cross_model_proposal * n_cross_model_proposals
# run jax-based

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

from bayes_implicit_solvent.samplers import random_walk_mh

for chunk in trange:
    tree = make_one_rjmc_proposal(tree_traj[-1])
    types = tree.apply_to_molecule_list(oemols)
    theta0 = np.hstack([tree.get_radii(), tree.get_scale_factors()])

    N = int(len(theta0) / 2)
    stepsize = np.ones(N * 2)
    stepsize[:N] = 0.0005
    stepsize[N:] = 0.001
    if ll == 'gaussian':
        stepsize *= 0.25

    def log_prob_fun(theta):
        predictions = get_predictions(theta, types)
        return log_prior(theta) + log_likelihood_of_predictions(predictions)



    mh_result = random_walk_mh(theta0, log_prob_fun, n_steps=n_within_model_steps_per_cross_model_proposal, stepsize=stepsize)

    theta = mh_result[0][-1]
    tree.set_radii(theta[:N])
    tree.set_scale_factors(theta[N:])

    tree_traj.append(tree)
    n_types_traj.append(N)
    within_model_trajs.append(mh_result[0])

    trange.set_postfix(max_n_types=max(n_types_traj), min_n_types=min(n_types_traj))

    np.savez('elaborate_tree_rjmc_march29_run_n_compounds={}_n_iter={}_{}_ll.npz'.format(len(mols), n_iterations, ll),
             n_types_traj=n_types_traj,
             within_model_trajs=within_model_trajs,
             )

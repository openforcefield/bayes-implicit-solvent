from numpy import load, random
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule

import sys

valid_lls = ['student-t', 'gaussian']

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
random.seed(job_id)
random.shuffle(paths_to_samples)
# 90:10 split

paths_to_samples = paths_to_samples[:int(0.9*len(paths_to_samples))]

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

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp

@jit
def predict_solvation_free_energy_jax(radii, scaling_factors, distance_matrices, charges):

    @jit
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)

    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)

distance_matrices = [mol.distance_matrices for mol in mols]
charges = [mol.charges for mol in mols]

expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncs = np.array([mol.experimental_uncertainty for mol in mols])

from bayes_implicit_solvent.typers import RADIUS_UNIT

from bayes_implicit_solvent.typers import AtomSpecificationProposal

from bayes_implicit_solvent.typers import GBTypingTree
initial_tree = GBTypingTree(max_nodes=20)
initial_tree.proposal_sigmas['radius'] = 1e-2 * RADIUS_UNIT
initial_tree.proposal_sigmas['scale_factor'] = 1e-2

initial_tree.add_child('[*]', '*') # TODO: fix this stupidity, by changing all references to '*' as root to '[*]' as root...
initial_tree.un_delete_able_types.add('[*]')
#initial_tree.is_decorate_able = lambda node: True

atomic_numbers = [1,6,7,8,9,15,16,17,35,53]

atomic_number_specifiers = ['#{}'.format(n) for n in atomic_numbers]
ring_specifiers = ['r0', 'r3', 'r4', 'r5', 'r6', 'r7', 'a', 'A']
charge_specifiers = ['-1', '+0', '+1', '+2']
hydrogen_count_specifiers = ['H0', 'H1', 'H2', 'H3', 'H4']
connectivity_specifiers = ['X1', 'X2', 'X3', 'X4']

#specifiers = connectivity_specifiers
specifiers = atomic_number_specifiers + ring_specifiers + charge_specifiers + hydrogen_count_specifiers + connectivity_specifiers
#primitives = list(map(lambda s: '[{}]'.format(s), specifiers))

from bayes_implicit_solvent.typers import DiscreteProposal

#smirks_elaboration_proposal = RootSpecificationProposal(primitives)
atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
smirks_elaboration_proposal = atom_specification_proposal

print('initial tree:')
print(initial_tree)

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
    N = int(len(theta) / 2)
    radii_, scaling_factors_ = theta[:N], theta[N:]

    radii = [radii_[types[i]] for i in range(len(types))]
    scaling_factors = [scaling_factors_[types[i]] for i in range(len(types))]
    return np.array([predict_solvation_free_energy_jax(radii[i], scaling_factors[i], distance_matrices[i], charges[i]) for i in range(len(charges))])


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

n_within_model_steps_per_cross_model_proposal = 1
n_cross_model_proposals = 10000
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

from bayes_implicit_solvent.samplers import random_walk_mh

train_smiles = [mol.smiles for mol in mols]

def save():
    name = 'tree_rjmc_from_wildcard_n_iter={}_ll={}_job_id={}'.format(
        n_iterations,
        ll,
        job_id
    )
    np.savez(name + '.npz',
             ll=ll,
             job_id=job_id,
             train_smiles=np.array(train_smiles),
             n_types_traj=n_types_traj,
             within_model_trajs=within_model_trajs,
             expt_means=expt_means,
             expt_uncs=expt_uncs,
             )

    from pickle import dump
    with open(name + '.pkl', 'wb') as f:
        dump(tree_traj, f)

from copy import deepcopy

for chunk in trange:
    tree = make_one_rjmc_proposal(tree_traj[-1])
    types = tree.apply_to_molecule_list(oemols)
    theta0 = np.hstack([tree.get_radii(), tree.get_scale_factors()])

    N = int(len(theta0) / 2)
    stepsize = np.ones(N * 2)
    # TODO: Improve heuristic... / Replace with MALA / HMC...
    stepsize[:N] = 0.005 / 10
    stepsize[N:] = 0.01 / 10
    if ll == 'gaussian':
        stepsize *= 0.25

    def log_prob_fun(theta):
        predictions = get_predictions(theta, types)
        return log_prior(theta) + log_likelihood_of_predictions(predictions)

    mh_result = random_walk_mh(theta0, log_prob_fun, n_steps=n_within_model_steps_per_cross_model_proposal, stepsize=stepsize)

    theta = mh_result[0][-1]
    tree.set_radii(theta[:N])
    tree.set_scale_factors(theta[N:])

    tree_traj.append(deepcopy(tree))
    n_types_traj.append(N)
    for t in mh_result[0]:
        within_model_trajs.append(t)

    trange.set_postfix(max_n_types=max(n_types_traj), min_n_types=min(n_types_traj))

    if (chunk + 1) % 500 == 0:
        save()

save()


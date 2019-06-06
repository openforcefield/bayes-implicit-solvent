import numpy as np

from bayes_implicit_solvent.continuous_parameter_experiments.elemental_types_mh import log_prior, mols, ll, data_path, \
    smiles

smiles_list = smiles
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

n_configuration_samples = 25

import os

name = 'tree_rjmc_n_config={}_{}_ll'.format(n_configuration_samples, ll)
smiles_subset_fname = os.path.join(data_path,
                                   'smiles_subset_{}.txt'.format(name))
with open(smiles_subset_fname, 'w') as f:
    f.writelines(['{}\n'.format(s) for s in smiles_list])

from bayes_implicit_solvent.prior_checking import check_no_empty_types

error_y_trees = []


def log_prob(tree):
    log_prior_value = check_no_empty_types(tree)

    theta = np.hstack([tree.get_radii(), tree.get_scale_factors()])

    log_prior_value += log_prior(theta)

    if log_prior_value > -np.inf:
        try:
            # TODO: Parallelize. Note that multiprocessing.Pool won't work here because it doesn't play nice with SwigPy objects
            # TODO: update to allow scale factors to be variable also
            log_likelihood_value = 0
            for mol in mols:
                radii = tree.assign_radii(mol.mol) / RADIUS_UNIT
                scale_factors = tree.assign_scale_factors(mol.mol)

                log_likelihood_value += mol.log_prob(radii, scale_factors)
        except:
            global error_y_trees
            error_y_trees.append(tree)
            print('Warning! Encountered un-anticipated exception!')
            return - np.inf
        return log_prior_value + log_likelihood_value
    else:
        return log_prior_value


from bayes_implicit_solvent.samplers import tree_rjmc
from pickle import dump

import itertools
n_iterations_per_chunk = 1000
n_chunks = 100
n_iterations = n_chunks * n_iterations_per_chunk
# run jax-based

trajs = []
log_prob_trajs = []
log_acceptance_probabilities_trajs = []
n_types_trajs = []
proposal_dim_trajs = []

from tqdm import tqdm

trange = tqdm(range(n_chunks))
for chunk in trange:
    result = tree_rjmc(initial_tree, log_prob, smirks_elaboration_proposal, n_iterations=n_iterations_per_chunk,
                       fraction_cross_model_proposals=0.25)

    trajs.append(result["traj"][1:])
    initial_tree = trajs[-1][-1]
    log_prob_trajs.append(result["log_probs"][1:])
    log_acceptance_probabilities_trajs.append(result["log_acceptance_probabilities"])
    n_types_trajs.append(np.array([tree.number_of_nodes for tree in result['traj']]))
    proposal_dim_trajs.append(result['proposal_move_dimensions'])

    traj = list(itertools.chain(*trajs))
    log_probs = np.hstack(log_prob_trajs)
    log_acceptance_probabilities = np.hstack(log_acceptance_probabilities_trajs)
    n_types_traj = np.hstack(n_types_trajs)
    proposal_dims = np.hstack(proposal_dim_trajs)


    trange.set_postfix(max_n_types=max(n_types_traj), min_n_types=min(n_types_traj))

    np.savez('elaborate_tree_rjmc3_run_n_compounds={}_n_iter={}_{}_ll.npz'.format(len(mols), n_iterations, ll),
             n_types_traj=n_types_traj,
             log_probs=log_probs,
             log_accept_prob_traj=log_acceptance_probabilities,
             proposal_dims=proposal_dims,
             tree_strings=[str(tree) for tree in traj],
             radii=[tree.get_radii() for tree in traj],
             scales=[tree.get_scale_factors() for tree in traj],
             )

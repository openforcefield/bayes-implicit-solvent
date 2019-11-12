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
initial_tree.proposal_sigmas['radius'] = 1e-2 * RADIUS_UNIT
initial_tree.proposal_sigmas['scale_factor'] = 1e-2

# add one more parameter per element appearing in FreeSolv but not specified in obc2 parameter set to initial tree
for i in [17, 35, 53]:
    smirks = '[#{}]'.format(i)
    initial_tree.add_child(smirks, '*')
    initial_tree.un_delete_able_types.add(smirks)

#specifiers = ['X1', 'X2', 'X3', 'X4', 'a', 'A']
#atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
#smirks_elaboration_proposal = atom_specification_proposal



all_bond_specifiers = ['@', '-', '#', '=', ':']
from bayes_implicit_solvent.smarts import atomic_number_dict

all_bondable_types = list(atomic_number_dict.keys())

# atomic_decorators list:
ring_specifiers = ['r0', 'r3', 'r4', 'r5', 'r6', 'r7', 'a', 'A']
charge_specifiers = ['-1', '+0', '+1', '+2']
hydrogen_count_specifiers = ['H0', 'H1', 'H2', 'H3', 'H4']
connectivity_specifiers = ['X1', 'X2', 'X3', 'X4']

all_specifier_lists = [
    ring_specifiers,
    charge_specifiers,
    hydrogen_count_specifiers,
    connectivity_specifiers,
]

from itertools import chain


all_atomic_specifiers = list(chain(*all_specifier_lists))
#all_bondable_types += ['[{}]'.format(s) for s in all_atomic_specifiers]
#all_decorators = all_bondable_types + all_atomic_specifiers + all_bond_specifiers


from bayes_implicit_solvent.typers import BondProposal, BondSpecificationProposal, AtomSpecificationProposal, SMIRKSElaborationProposal
#bond_proposal = BondProposal(bondable_types=all_bondable_types)
atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=all_atomic_specifiers)
#bond_specification_proposal = BondSpecificationProposal(bond_specifiers=all_bond_specifiers)

smirks_elaborators = [
    #bond_proposal,
    atom_specification_proposal,
    #bond_specification_proposal,
]
smirks_elaboration_proposal = SMIRKSElaborationProposal(smirks_elaborators=smirks_elaborators)

print('initial tree:')
print(initial_tree)

n_configuration_samples = 5

import os

name = 'tree_rjmc_n_config={}_{}_ll'.format(n_configuration_samples, ll)
smiles_subset_fname = os.path.join(data_path,
                                   'smiles_subset_{}.txt'.format(name))
with open(smiles_subset_fname, 'w') as f:
    f.writelines(['{}\n'.format(s) for s in smiles_list])

from bayes_implicit_solvent.prior_checking import check_no_empty_types

error_y_trees = []

for mol in mols:
    thinning = int(len(mol.vacuum_traj) / n_configuration_samples)
    mol.vacuum_traj = mol.vacuum_traj[::thinning]


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

n_iterations = 10000

result = tree_rjmc(initial_tree, log_prob, smirks_elaboration_proposal, n_iterations=n_iterations,
                   fraction_cross_model_proposals=0.1)
with open('elaborate_tree_rjmc_run_n_compounds={}_n_iter={}_gaussian_ll.pkl'.format(len(mols), n_iterations),
          'wb') as f:
    dump(result, f)

with open('error_y_trees.pkl', 'wb') as f:
    dump(error_y_trees, f)

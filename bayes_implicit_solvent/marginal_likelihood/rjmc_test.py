# set up an RJMC simulation that should visit only the states with (*) or (*, [#6]) as types...

import numpy as np

from bayes_implicit_solvent.typers import GBTypingTree
from bayes_implicit_solvent.typers import RADIUS_UNIT

initial_tree = GBTypingTree()
initial_tree.proposal_sigmas['radius'] = 1e-2 * RADIUS_UNIT
initial_tree.proposal_sigmas['scale_factor'] = 1e-2

from bayes_implicit_solvent.typers import AtomSpecificationProposal, SMIRKSElaborationProposal

atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=['[#6]'])
smirks_elaborators = [
    atom_specification_proposal,
]
smirks_elaboration_proposal = SMIRKSElaborationProposal(smirks_elaborators=smirks_elaborators)

print('initial tree:')
print(initial_tree)

from bayes_implicit_solvent.marginal_likelihood.single_type_forward_ais import log_posterior as log_posterior_one_type
from bayes_implicit_solvent.marginal_likelihood.two_types_radius_and_scale_forward_ais import \
    log_posterior as log_posterior_two_types
from bayes_implicit_solvent.marginal_likelihood.single_type_forward_ais import mols


def log_posterior(theta):
    if len(theta) == 2:
        return log_posterior_one_type(theta)
    elif len(theta) == 4:
        return log_posterior_two_types(theta)
    else:
        raise (RuntimeError())


def log_prob(tree):
    theta = np.hstack([tree.get_radii(), tree.get_scale_factors()])
    return log_posterior(theta)


from bayes_implicit_solvent.samplers import tree_rjmc
from pickle import dump

n_iterations = 10000

result = tree_rjmc(initial_tree, log_prob, smirks_elaboration_proposal, n_iterations=n_iterations,
                   fraction_cross_model_proposals=0.1)
with open('simple_tree_rjmc_run_n_compounds={}_n_iter={}_gaussian_ll.pkl'.format(len(mols), n_iterations),
          'wb') as f:
    dump(result, f)

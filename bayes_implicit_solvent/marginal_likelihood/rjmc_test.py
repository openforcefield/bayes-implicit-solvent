# set up an RJMC simulation that should visit only the states with (*) or (*, [#6]) as types...

import numpy as np

from bayes_implicit_solvent.typers import GBTypingTree
from bayes_implicit_solvent.typers import RADIUS_UNIT

from bayes_implicit_solvent.typers import DiscreteProposal
class RootSpecificationProposal(DiscreteProposal):
    def __init__(self, elaborations):
        self.elaborations = elaborations

    def log_prob_forward(self, initial_smirks, proposed_smirks=None):
        return - np.log(len(self.elaborations)) # since I can only elaborate on the root

    def sample(self, initial_smirks):
        assert(initial_smirks == '*')
        proposal_smirks = self.elaborations[np.random.randint(len(self.elaborations))]

        return {
            'proposal': proposal_smirks,
            'log_prob_forward_over_reverse': self.log_prob_forward_over_reverse(initial_smirks, proposal_smirks),
        }

smirks_elaboration_proposal = RootSpecificationProposal(['[#6]'])


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

n_iterations = 10000
proposal_sigmas = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
n_types_trajs = []
tree_trajs = []
log_prob_trajs = []
log_accept_prob_trajs = []
for proposal_sigma in proposal_sigmas:
    initial_tree = GBTypingTree(max_nodes=2)
    initial_tree.proposal_sigmas['radius'] = proposal_sigma * RADIUS_UNIT
    initial_tree.proposal_sigmas['scale_factor'] = proposal_sigma
    initial_tree.is_decorate_able = lambda \
        node: node == '*'  # overwrite method that currently returns False if the node is the root node...

    # TODO: clue: the amount of time it spends in each model may be related to proposal sigmas
    # TODO: clue: I see the progress bar is saying `avg. accept. prob.=nan` but it hasn't resulted in an error!
    # initial_tree.add_child('[#6]', '*')
    print('initial tree:')
    print(initial_tree)

    result = tree_rjmc(initial_tree, log_prob, smirks_elaboration_proposal, n_iterations=n_iterations,
                       fraction_cross_model_proposals=0.25)
    traj = result["traj"]
    log_probs = result["log_probs"]
    log_acceptance_probabilities = result["log_acceptance_probabilities"]

    n_types_trajs.append(np.array([tree.number_of_nodes for tree in result['traj']]))
    tree_trajs.append(traj)
    log_prob_trajs.append(log_probs)
    log_accept_prob_trajs.append(log_acceptance_probabilities)

    print('proposal_sigma={}'.format(proposal_sigma))
    print('time spent in each model: n=1: {}, n=2: {}'.format(sum(n_types_trajs[-1] == 1), sum(n_types_trajs[-1] == 2)))

np.savez('rjmc_test.npz',
         proposal_sigmas=proposal_sigmas,
         n_types_trajs=n_types_trajs,
         log_prob_trajs=log_prob_trajs,
         log_accept_prob_trajs=log_accept_prob_trajs,
         tree_strings=[[str(tree) for tree in traj] for traj in tree_trajs],
         radii=[[tree.get_radii() for tree in traj] for traj in tree_trajs],
         scales=[[tree.get_scale_factors() for tree in traj] for traj in tree_trajs],
         )

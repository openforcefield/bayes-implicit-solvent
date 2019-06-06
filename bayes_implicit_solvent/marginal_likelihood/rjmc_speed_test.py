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


prior_location = np.array([0.15, 0.8])
radius_lower_bound = 0.01
scale_lower_bound = 0.1

from scipy.stats import norm
Z_r = (1 - norm.cdf(radius_lower_bound, loc=prior_location[0]))
Z_s = (1 - norm.cdf(scale_lower_bound, loc=prior_location[1]))
log_Z_prior_one_type = np.log((Z_r * Z_s))
log_Z_prior_two_types = np.log((Z_r * Z_s)**2)

AIS_guess_free_energy_difference = 3.2675 # current best guess of - (log_Z_posterior_2 - log_Z_posterior_1)
import itertools

def log_posterior(theta):
    if len(theta) == 2:
        return log_posterior_one_type(theta) - log_Z_prior_one_type
    elif len(theta) == 4:
        # TODO: Revisit if I want to include a bias here: currently adding in my current guess for the free energy difference
        # TODO: which I hope will approximately flatten sampling between the two models...
        return log_posterior_two_types(theta) - log_Z_prior_two_types + AIS_guess_free_energy_difference
    else:
        raise (RuntimeError())


def log_prob(tree):
    theta = np.hstack([tree.get_radii(), tree.get_scale_factors()])
    return log_posterior(theta)


from bayes_implicit_solvent.samplers import tree_rjmc

n_iterations_per_chunk = 1000
n_chunks = 100
proposal_sigmas = np.linspace(1e-3, 1e-1, 10)
inds = np.arange(len(proposal_sigmas))
np.random.shuffle(inds)


if __name__ == "__main__":


    for i in inds:
        proposal_sigma = proposal_sigmas[i]
        initial_tree = GBTypingTree(max_nodes=2)
        initial_tree.proposal_sigmas['radius'] = proposal_sigma * RADIUS_UNIT
        initial_tree.proposal_sigmas['scale_factor'] = proposal_sigma
        initial_tree.is_decorate_able = lambda \
            node: node == '*'  # overwrite method that currently returns False if the node is the root node...

        # TODO: clue: the amount of time it spends in each model may be related to proposal sigmas
        # TODO: inspect what would be the average cross-model acceptance probability throughout the trajectory? (side-effect: get a better estimate of the ratio of marginal likelihoods from the simulation output...)
        # TODO: figure out why iterations / second decreases so much over time for tree_rjmc (it completes iterations 1 through 200 in under a second, and iterations 9800 through 10000 in almost 30 seconds...)
        # initial_tree.add_child('[#6]', '*')
        print('initial tree:')
        print(initial_tree)

        trajs = []
        log_prob_trajs = []
        log_acceptance_probabilities_trajs = []
        n_types_trajs = []
        proposal_dim_trajs = []

        for chunk in range(n_chunks):
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

        print('proposal_sigma={}'.format(proposal_sigma))
        print('time spent in each model: n=1: {}, n=2: {}'.format(sum(n_types_traj == 1), sum(n_types_traj == 2)))

        np.savez('rjmc_test_coarse_grid/proposal_sigmas[{}].npz'.format(i),
                 proposal_sigma=proposal_sigma,
                 n_types_traj=n_types_traj,
                 log_probs=log_probs,
                 log_accept_prob_traj=log_acceptance_probabilities,
                 proposal_dims=proposal_dims,
                 tree_strings=[str(tree) for tree in traj],
                 radii=[tree.get_radii() for tree in traj],
                 scales=[tree.get_scale_factors() for tree in traj],
                 )

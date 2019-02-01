# JDC suggests two tests:
# * Likelihood always zero, confirm that we get uniform distribution
# * Likelihood Gaussian in GB radius, confirm that we get expected result

import numpy as np
import pytest
from networkx import nx

from bayes_implicit_solvent.samplers import tree_rjmc
from bayes_implicit_solvent.typers import GBTypingTree, AtomSpecificationProposal


def assert_self_consistency(initial_tree, max_tries=100):
    """Up to max_tries times, sample the creation proposal, then sample the deletion proposal, and
    if you get back initial tree, confirm that log_prob_forward_over_reverse is consistent in the two directions"""

    for _ in range(max_tries):
        elaborated_proposal = initial_tree.sample_creation_proposal()
        elaborate_tree = elaborated_proposal['proposal']
        pruned_proposal = elaborate_tree.sample_deletion_proposal()
        pruned_tree = pruned_proposal['proposal']

        if (tuple(pruned_tree.ordered_nodes) == tuple(initial_tree.ordered_nodes)):
            f = elaborated_proposal['log_prob_forward_over_reverse']
            r = - pruned_proposal['log_prob_forward_over_reverse']

            if not np.isclose(f, r):
                pytest.fail('Inconsistent pair detected \n\t{}\n\t{}'.format(elaborated_proposal, pruned_proposal))
            else:
                return True
    print(RuntimeWarning(
        "Wasn't able to make a reversible pair of jumps in {} attempts for\n{}".format(max_tries, initial_tree)))


def construct_initial_tree():
    """Construct a basic tree with a hydrogen and the ability to specify connectivity"""
    specifiers = ['X1', 'X2', 'X3', 'X4']
    atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
    un_delete_able_types = ['*', '[#1]', '[#2]']
    initial_tree = GBTypingTree(smirks_elaboration_proposal=atom_specification_proposal,
                                un_delete_able_types=un_delete_able_types)
    for base_type in un_delete_able_types[1:]:
        initial_tree.add_child(child_smirks=base_type, parent_smirks='*')
    return initial_tree


def test_proposal_self_consistency_on_random_walk(walk_length=100):
    """Sample a sequence of elaborate trees, then evaluate the self-consistency of create/delete
    proposals for each tree visited in this sequence"""
    print('attempting a random walk')
    traj = [construct_initial_tree()]
    for _ in range(walk_length):
        traj.append(traj[-1].sample_creation_proposal()['proposal'])
    for tree in traj:
        assert_self_consistency(tree)


def test_atom_specification_proposal(n_tests=50):
    np.random.seed(0)

    initial_tree = construct_initial_tree()
    # adding and removing a single specifier
    for _ in range(n_tests):
        assert_self_consistency(initial_tree)

    print('depth-1 trees okay')
    # adding and removing more than one specifier
    for _ in range(n_tests):
        elaborated_proposal = initial_tree.sample_creation_proposal()['proposal']
        assert_self_consistency(elaborated_proposal)
    print('depth-2 trees okay')


from simtk import unit
from scipy.stats import multivariate_normal

@pytest.mark.slow
def test_uniform_sampling(depth_cutoff=2, n_iterations=100000):
    """Test that a sampler targeting these discrete structures and associated continuous parameters jointly
    obtains a uniform distribution over bounded-depth trees when appropriate.
    To do this, we ensure that each discrete tree has the same normalizing constant (namely, 1).

    # TODO: Choice of continuous distribution is arbitrary, as long as normalized. May switch to uniform instead of Gaussian.
    """

    np.random.seed(0)

    # specifiers = ['X1', 'X2', 'X3']
    specifiers = ['X1', 'X2']
    # specifiers = ['X1']

    # TODO: Set up testing fixtures with different numbers of specifiers, depth_cutoffs, etc.

    atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
    N = len(atom_specification_proposal.atomic_specifiers)
    un_delete_able_types = ['*', '[#1]']
    initial_tree = GBTypingTree(smirks_elaboration_proposal=atom_specification_proposal,
                                un_delete_able_types=un_delete_able_types,
                                proposal_sigma=1.0 * unit.nanometer,
                                )

    for base_type in un_delete_able_types[1:]:
        initial_tree.add_child(child_smirks=base_type, parent_smirks='*')

    from math import factorial
    n_trees_at_length = lambda length: int(factorial(N) / factorial(N - length))

    number_of_trees_at_each_length = list(map(n_trees_at_length, range(len(specifiers) + 1)))

    def log_prob(tree):
        """To induce a uniform marginal distribution over *discrete* trees
        up to depth cutoff without duplicated nodes:
        1. check that the discrete tree is valid -- if not, return a log-probability of -inf
        2. define a normalized distribution over each tree's *continuous* parameters,
        namely a multivariate normal distribution

        If we sample the resulting probability distribution properly, we should obtain:
        1. A uniform marginal distribution over valid discrete trees
        2. A gaussian distribution over the continuous parameters within each model
        """
        N_nodes = tree.number_of_nodes
        no_duplicates = (len(set(tree.nodes)) == N_nodes)
        within_depth_limit = (max(nx.shortest_path_length(tree.G, source='*').values()) <= depth_cutoff)
        if no_duplicates and within_depth_limit:
            mean_vector = np.ones(N_nodes) # a length-N_nodes vector of 1's
            tree_radii = tree.get_radii() # a length-N_nodes vector of the radii associated with nodes in the tree
            return multivariate_normal.logpdf(x=tree_radii, mean=mean_vector)
        else:
            return - np.inf
    np.random.seed(0)
    result = tree_rjmc(initial_tree, log_prob,
                       n_iterations=n_iterations,
                       fraction_cross_model_proposals=0.5,
                       )
    radii = [tree.get_radii() for tree in result['traj']]

    np.save('sampled_radii.npy', radii)

    print('number of possible distinct discrete trees at each length',
          list(zip(range(len(number_of_trees_at_each_length)), number_of_trees_at_each_length)))

    number_of_possibilities = sum(number_of_trees_at_each_length)
    print('number of possibilities:', number_of_possibilities)

    print('initial tree:')
    print(initial_tree)

    traj = result['traj']
    string_representations = list(map(str, traj))
    print('number of distinct sampled models (as reflected in string representation)', len(set(string_representations)))
    for s in list(set(string_representations))[:5]:
        print(s)

    discrete_models = [tuple(t.ordered_nodes[2:]) for t in traj]
    distinct_discrete_models = sorted(list(set(discrete_models)))
    for d in distinct_discrete_models:
        print(d)
    print("number of distinct sampled models (as reflected in choice of smirks)", len(distinct_discrete_models))
    thinning = 20
    lengths = np.array([len(d) for d in discrete_models[::thinning]])

    expected_length_distribution = len(lengths) * (np.array(number_of_trees_at_each_length) / np.sum(number_of_trees_at_each_length))
    actual_length_distribution = np.zeros(len(expected_length_distribution))
    for t in range(len(expected_length_distribution)):
        actual_length_distribution[t] += sum(lengths == t)
    print('expected_length_distribution', expected_length_distribution)
    print('actual_length_distribution', actual_length_distribution)

    threshold = 0.001

    from scipy.stats import chisquare
    chi2_result = chisquare(f_obs=actual_length_distribution, f_exp=expected_length_distribution)
    print(chi2_result)
    assert (chi2_result.pvalue > threshold)

    from scipy.stats import kstest

    for i in range(max(lengths)):
        rvs = np.array([r[i] for r in radii if len(r) > i])

        # check that we're not producing mean-zero Gaussian values
        kstest_result = kstest(rvs[::thinning], 'norm')
        pvalue_should_be_under_threshold = kstest_result.pvalue


        assert (pvalue_should_be_under_threshold < threshold)

        # check that we're producing mean 1.0 Gaussian values
        from scipy.stats import norm
        kstest_result = kstest(rvs[::thinning], norm(loc=1.0).cdf)
        pvalue_should_be_over_threshold = kstest_result.pvalue

        assert (pvalue_should_be_over_threshold > threshold)

    return result

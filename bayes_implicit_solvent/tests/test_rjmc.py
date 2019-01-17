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


def test_uniform_sampling(depth_cutoff=2, n_iterations=50000):
    """Test that we get a uniform distribution over bounded-depth trees"""

    np.random.seed(0)

    # specifiers = ['X1', 'X2', 'X3']
    specifiers = ['X1']

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
        """Uniform distribution over trees up to depth cutoff without duplicated nodes"""

        if (len(set(tree.nodes)) == tree.number_of_nodes) and \
                (max(nx.shortest_path_length(tree.G, source='*').values()) <= depth_cutoff):
            # return 0
            return np.sum(
                [((initial_tree.get_radius(smirks) - (0.1 * unit.nanometer)) / unit.nanometer) ** 2 for smirks in
                 initial_tree.nodes])
        else:
            return - np.inf

    result = tree_rjmc(initial_tree, log_prob,
                       n_iterations=n_iterations,
                       fraction_cross_model_proposals=0.5,
                       )

    print('number of possible distinct discrete trees at each length',
          list(zip(range(len(number_of_trees_at_each_length)), number_of_trees_at_each_length)))

    number_of_possibilities = sum(number_of_trees_at_each_length)
    print('number of possibilities:', number_of_possibilities)

    print('initial tree:')
    print(initial_tree)

    traj = result['traj']
    string_representations = list(map(str, traj))
    print('number of distinct sampled models (as reflected in string representation)', len(set(string_representations)))
    for s in sorted(list(set(string_representations))):
        print(s)

    discrete_models = [tuple(t.ordered_nodes[2:]) for t in traj]
    distinct_discrete_models = sorted(list(set(discrete_models)))
    for d in distinct_discrete_models:
        print(d)
    print("number of distinct sampled models (as reflected in choice of smirks)", len(distinct_discrete_models))

    lengths = np.array([len(d) for d in discrete_models])

    expected_length_distribution = np.array(number_of_trees_at_each_length) / np.sum(number_of_trees_at_each_length)
    actual_length_distribution = np.zeros(len(expected_length_distribution))
    for t in range(len(expected_length_distribution)):
        actual_length_distribution[t] += sum(lengths == t)
    actual_length_distribution /= np.sum(actual_length_distribution)
    print('expected_length_distribution', expected_length_distribution)
    print('actual_length_distribution', actual_length_distribution)

    assert (np.allclose(expected_length_distribution, actual_length_distribution, rtol=1e-2))
    return result

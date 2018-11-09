# JDC suggests two tests:
# * Likelihood always zero, confirm that we get uniform distribution
# * Likelihood Gaussian in GB radius, confirm that we get expected result

import numpy as np

from bayes_implicit_solvent.typers import GBTypingTree, AtomSpecificationProposal, BondSpecificationProposal, BondProposal

from networkx import nx

from bayes_implicit_solvent.samplers import tree_rjmc

import pytest

def test_atom_specification_proposal(n_trials=100):
    np.random.seed(0)
    specifiers = ['X1', 'X2', 'X3', 'X4']
    atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
    un_delete_able_types = ['*', '[#1]']
    initial_tree = GBTypingTree(smirks_elaboration_proposal=atom_specification_proposal,
                                un_delete_able_types=un_delete_able_types)
    initial_tree.add_child(child_smirks=un_delete_able_types[1], parent_smirks='*')

    # adding and removing a single specifier
    for _ in range(n_trials):
        elaborated_proposal = initial_tree.sample_creation_proposal()
        elaborate_tree = elaborated_proposal['proposal']
        pruned_proposal = elaborate_tree.sample_deletion_proposal()

        f = elaborated_proposal['log_prob_forward_over_reverse']
        r = - pruned_proposal['log_prob_forward_over_reverse']

        if not np.isclose(f, r):
            pytest.fail('Inconsistent pair detected \n\t{}\n\t{}'.format(elaborated_proposal, pruned_proposal))
    print('depth-1 trees okay')
    # adding and removing more than one specifier
    for _ in range(n_trials):
        elaborated_proposal = initial_tree.sample_creation_proposal()
        elaborate_tree = elaborated_proposal['proposal']
        twice_elaborated_proposal = elaborate_tree.sample_creation_proposal()
        twice_elaborated_tree = twice_elaborated_proposal['proposal']
        pruned_proposal = twice_elaborated_tree.sample_deletion_proposal()
        pruned_tree = pruned_proposal['proposal']

        if (tuple(pruned_tree.ordered_nodes) == tuple(elaborate_tree.ordered_nodes)):
            f = twice_elaborated_proposal['log_prob_forward_over_reverse']
            r = - pruned_proposal['log_prob_forward_over_reverse']
            if not np.isclose(f, r):
                pytest.fail('Inconsistent pair detected \n\t{}\n\t{}'.format(twice_elaborated_proposal, pruned_proposal))
    print('depth-2 trees okay')

def test_uniform_sampling(depth_cutoff=2, n_iterations=10000):
    """Test that we get a uniform distribution over bounded-depth trees"""

    np.random.seed(0)

    specifiers = ['X1', 'X2', 'X3', 'X4']

    atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)

    un_delete_able_types = ['*', '[#1]']
    initial_tree = GBTypingTree(smirks_elaboration_proposal=atom_specification_proposal,
                                un_delete_able_types=un_delete_able_types)
    for base_type in un_delete_able_types[1:]:
        initial_tree.add_child(child_smirks=base_type, parent_smirks='*')

    from math import factorial
    n_trees_at_length = lambda length : int(factorial(len(specifiers)) / factorial(len(specifiers) - length))

    number_of_trees_at_each_length = list(map(n_trees_at_length, range(len(specifiers) + 1)))

    def log_prob(tree):
        """Uniform distribution over trees up to depth cutoff without duplicated nodes"""

        if (len(set(tree.nodes)) == tree.number_of_nodes) and \
                (max(nx.shortest_path_length(tree.G, source='*').values()) <= depth_cutoff):
            return 0
        else:
            return - np.inf

    result = tree_rjmc(initial_tree, log_prob,
                       n_iterations=n_iterations,
                       fraction_cross_model_proposals=0.99)

    print('number of possible distinct discrete trees at each length', list(zip(range(len(number_of_trees_at_each_length)), number_of_trees_at_each_length)))

    number_of_possibilities = sum(number_of_trees_at_each_length)
    print('number of possibilities:', number_of_possibilities)

    print('initial tree:')
    print(initial_tree)

    traj = result['traj']
    discrete_models = [tuple(t.ordered_nodes[2:]) for t in traj]
    distinct_discrete_models = sorted(list(set(discrete_models)))
    for d in distinct_discrete_models:
        print(d)
    print("Number of distinct sampled models:", len(distinct_discrete_models))

    lengths = np.array([len(d) for d in discrete_models])

    expected_length_distribution = np.array(number_of_trees_at_each_length) / np.sum(number_of_trees_at_each_length)
    actual_length_distribution = np.zeros(len(expected_length_distribution))
    for t in range(len(expected_length_distribution)):
        actual_length_distribution[t] += sum(lengths == t)
    actual_length_distribution /= np.sum(actual_length_distribution)
    print('expected_length_distribution', expected_length_distribution)
    print('actual_length_distribution', actual_length_distribution)

    assert(np.allclose(expected_length_distribution, actual_length_distribution, rtol=1e-2))
    return result


if __name__ == "__main__":
    test_atom_specification_proposal()
    #result = test_uniform_sampling()

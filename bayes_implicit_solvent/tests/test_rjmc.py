# JDC suggests two tests:
# * Likelihood always zero, confirm that we get uniform distribution
# * Likelihood Gaussian in GB radius, confirm that we get expected result

import numpy as np

from bayes_implicit_solvent.typers import GBTypingTree, AtomSpecificationProposal

from networkx import nx

from bayes_implicit_solvent.samplers import tree_rjmc


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

    number_of_trees_at_each_length = list(map(factorial, range(len(specifiers) + 1)))

    number_of_possibilities = sum(number_of_trees_at_each_length)
    print('number of possibilities:', number_of_possibilities)

    print('initial tree:')
    print(initial_tree)


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

    traj = result['traj']
    discrete_models = [tuple(t.ordered_nodes[2:]) for t in traj]
    distinct_discrete_models = sorted(list(set(discrete_models)))
    for d in distinct_discrete_models:
        print(d)
    print("Number of distinct sampled models:", len(distinct_discrete_models))
    discrete_model_dict = dict(zip(distinct_discrete_models, range(len(distinct_discrete_models))))
    desired_distribution = np.ones(number_of_possibilities) / number_of_possibilities
    actual_distribution = np.zeros(number_of_possibilities)
    for d in discrete_models:
        actual_distribution[discrete_model_dict[d]] += 1
    actual_distribution /= np.sum(actual_distribution)

    print(actual_distribution)

    lengths = np.array([len(d) for d in discrete_models])

    expected_length_distribution = np.array(number_of_trees_at_each_length) / np.sum(number_of_trees_at_each_length)
    actual_length_distribution = np.zeros(len(expected_length_distribution))
    for t in range(len(expected_length_distribution)):
        actual_length_distribution[t] += sum(lengths == t)
    actual_length_distribution /= np.sum(actual_length_distribution)

    return result, np.allclose(expected_length_distribution, actual_length_distribution, rtol=1e-2)


if __name__ == "__main__":
    result, length_distribution_okay = test_uniform_sampling()

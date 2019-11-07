from jax.config import config

config.update("jax_enable_x64", True)


def get_molecule_list():
    from numpy import load
    from simtk import unit

    from bayes_implicit_solvent.molecule import Molecule

    def sample_path_to_unitted_snapshots(path_to_npy_samples):
        xyz = load(path_to_npy_samples)
        traj = [snapshot * unit.nanometer for snapshot in xyz]
        return traj

    from glob import glob
    from pkg_resources import resource_filename

    path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                               'vacuum_samples/vacuum_samples_*.npy')
    paths_to_samples = glob(path_to_vacuum_samples)

    paths_to_samples = paths_to_samples

    print('number of molecules being considered: {}'.format(len(paths_to_samples)))

    def extract_cid_key(path):
        i = path.find('mobley_')
        j = path.find('.npy')
        return path[i:j]

    cids = list(map(extract_cid_key, paths_to_samples))
    mols = []

    n_configuration_samples = 25

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
            print(
                'discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles,
                                                                                                                cid))

    oemols = [mol.mol for mol in mols]
    return oemols


from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

tree = mbondi_model
tree.remove_node('[#14]')
tree.add_child('[#1]~[#8]', '[#1]')
tree.add_child('[#1]~[#6]', '[#1]')
oemols = get_molecule_list()
types = tree.apply_to_molecule_list(oemols)

from bayes_implicit_solvent.utils import Dataset

dataset = Dataset(oemols)

import numpy as onp


def get_types(tree):
    match_matrices = dataset.get_match_matrices(tree.smarts_list)
    return list(map(tree.assign_types_using_match_matrix, match_matrices))


def get_types_naive(tree):
    types = tree.apply_to_molecule_list(oemols)
    return types


def get_type_counts(tree):
    types = get_types(tree)
    counts = onp.bincount(onp.hstack(types), minlength=tree.number_of_nodes)
    return counts


def print_heaviest_types(tree):
    counts = get_type_counts(tree)
    inds = onp.argsort(-counts)
    for i in inds:
        print(tree.ordered_nodes[i], counts[i])


def get_probabilities_of_elaborating(tree):
    """elaborate on a type proportional to the number of atoms it hits,
    mask by whether elaborate-able
    """
    counts = get_type_counts(tree)
    decorate_able_mask = onp.array(list(map(tree.is_decorate_able, tree.ordered_nodes)))
    counts_ = counts * decorate_able_mask
    del (counts)
    return counts_ / onp.sum(counts_)


def get_probabilities_of_removing(tree):
    """delete a type inversely proportional to the number of atoms it hits,
    mask by whether delete-able (for example, only leaf nodes are ever delete-able)"""
    counts = get_type_counts(tree)
    delete_able_mask = onp.array(list(map(tree.is_delete_able, tree.ordered_nodes)))
    counts_ = counts * delete_able_mask
    del (counts)
    return counts_ / onp.sum(counts_)


# TODO: define a prior that favors not having overly specific types that only catch a very small number of
# TODO: atoms. One way to do this: dirichlet distribution with concentration parameter > 1.
from scipy.stats import dirichlet

def dirichlet_log_prior(counts, alpha=1.0):
    alpha_vector = onp.ones_like(counts) * alpha
    dp = dirichlet.logpdf(counts / counts.sum(), alpha=alpha_vector)
    correction = - dirichlet.logpdf(onp.ones_like(counts) / len(counts), alpha=alpha_vector)
    return dp + correction


if __name__ == "__main__":
    from bayes_implicit_solvent.typers import BondProposal, AtomSpecificationProposal, BondSpecificationProposal, \
        SMIRKSElaborationProposal
    from bayes_implicit_solvent.smarts import atomic_number_dict
    from bayes_implicit_solvent.prior_checking import NoEmptyTypesPrior

    no_empty_types_prior = NoEmptyTypesPrior(dataset)
    prior_alpha = 2.0

    def log_prior(tree):
        counts = get_type_counts(tree)
        return no_empty_types_prior.log_prob(tree) + dirichlet_log_prior(counts, prior_alpha)

    print('using the following decorators:')

    all_bond_specifiers = ['@', '-', '#', '=', ':']

    all_bondable_types = ['*'] + list(atomic_number_dict.keys())

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
    all_bondable_types += ['[{}]'.format(s) for s in all_atomic_specifiers]
    all_decorators = all_bondable_types + all_atomic_specifiers + all_bond_specifiers

    bond_proposal = BondProposal(bondable_types=all_bondable_types)
    atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=all_atomic_specifiers)
    bond_specification_proposal = BondSpecificationProposal(bond_specifiers=all_bond_specifiers)

    smirks_elaborators = [
        bond_proposal,
        atom_specification_proposal,
        bond_specification_proposal,
    ]
    smirks_elaboration_proposal = SMIRKSElaborationProposal(smirks_elaborators=smirks_elaborators)

    tree_traj = [tree]

    from tqdm import tqdm

    for _ in tqdm(range(100)):
        proposal_dict = tree_traj[-1].sample_create_delete_proposal(smirks_elaboration_proposal)
        tree, log_prob_forward_over_reverse = proposal_dict['proposal'], proposal_dict['log_prob_forward_over_reverse']
        if onp.exp(no_empty_types_prior.log_prob(tree) - log_prob_forward_over_reverse) > onp.random.rand():
            tree_traj.append(tree)
        else:
            tree_traj.append(tree_traj[-1])

    from copy import deepcopy


    # TODO: the arguments to this function can be (tree, dataset used to parameterize proposal, smirks_elaboration_proposal)
    def sample_informed_creation_proposal(tree):
        # Create a copy of the current typer, which we will modify and return
        proposal = deepcopy(tree)

        # sample a parent node according to elaboration-focusing proposal
        elaboration_probabilities = get_probabilities_of_elaborating(tree)
        node_index_to_elaborate = onp.random.choice(onp.arange(len(tree.ordered_nodes)), p=elaboration_probabilities)
        elaboration_probability = elaboration_probabilities[node_index_to_elaborate]
        parent_smirks = tree.ordered_nodes[node_index_to_elaborate]

        # create a new type by elaborating on the parent type
        elaboration_proposal_dict = smirks_elaboration_proposal.sample(parent_smirks)
        child_smirks = elaboration_proposal_dict['proposal']

        # create a new type by elaborating on the parent type
        proposal.add_child(child_smirks=child_smirks, parent_smirks=parent_smirks)

        # get probability of removing the child_smirks we just added according to deletion-focusing proposal
        removal_probabilities = get_probabilities_of_removing(proposal)
        index_of_child_smirks = proposal.ordered_nodes.index(child_smirks)
        removal_probability = removal_probabilities[index_of_child_smirks]

        # compute log probability of forward vs. reverse proposal...
        log_prob_forward_over_reverse = elaboration_proposal_dict['log_prob_forward_over_reverse']
        log_prob_forward_over_reverse += onp.log(elaboration_probability)
        log_prob_forward_over_reverse -= onp.log(removal_probability)

        proposal_dict = {
            'proposal': proposal,
            'log_prob_forward_over_reverse': log_prob_forward_over_reverse
        }

        return proposal_dict


    def sample_informed_deletion_proposal(tree):
        proposal = deepcopy(tree)

        # sample a leaf node according to deletion-focusing proposal
        removal_probabilities = get_probabilities_of_removing(tree)
        node_index_to_remove = onp.random.choice(onp.arange(len(tree.ordered_nodes)), p=removal_probabilities)
        node_to_remove = tree.ordered_nodes[node_index_to_remove]
        parent_of_removed_node = tree.get_parent_type(node_to_remove)
        removal_probability = removal_probabilities[node_index_to_remove]

        # remove the node we selected
        proposal.remove_node(node_to_remove)

        # get probability of adding the child smirks we just removed, according to elaboration-focusing proposal
        elaboration_probabilities = get_probabilities_of_elaborating(proposal)
        index_of_parent = proposal.ordered_nodes.index(parent_of_removed_node)
        elaboration_probability = elaboration_probabilities[index_of_parent]
        log_probability_to_choose_right_decorator = smirks_elaboration_proposal.log_prob_forward_over_reverse(
            parent_of_removed_node, node_to_remove)

        # compute log probability of forward vs. reverse proposal
        log_prob_forward_over_reverse = onp.log(removal_probability)
        log_prob_forward_over_reverse -= onp.log(elaboration_probability)
        log_prob_forward_over_reverse -= log_probability_to_choose_right_decorator

        proposal_dict = {
            'proposal': proposal,
            'log_prob_forward_over_reverse': log_prob_forward_over_reverse
        }

        return proposal_dict


    # TODO: also select the decorator proportional to split-evenness?
    # sample from prior by random walk...
    # TODO: add back continuous-parameter adjustment!
    focused_traj = [tree_traj[0]]
    log_prior_traj = [log_prior(focused_traj[0])]
    trange = tqdm(range(5000))
    for _ in trange:
        tree = focused_traj[-1]

        if tree.probability_of_sampling_a_create_proposal > onp.random.rand():
            proposal_dict = sample_informed_creation_proposal(tree)
        else:
            proposal_dict = sample_informed_deletion_proposal(tree)

        tree, log_prob_forward_over_reverse = proposal_dict['proposal'], proposal_dict['log_prob_forward_over_reverse']
        proposed_log_prior = log_prior(tree)

        if onp.exp(proposed_log_prior - log_prior_traj[-1] - log_prob_forward_over_reverse) > onp.random.rand():
            focused_traj.append(tree)
            log_prior_traj.append(proposed_log_prior)
        else:
            focused_traj.append(focused_traj[-1])
            log_prior_traj.append(log_prior_traj[-1])
        trange.set_postfix(log_prior=log_prior_traj[-1], n_types=focused_traj[-1].number_of_nodes)

    # TODO: plot distribution of n_types for a few choices of alpha: [0.5,1.0,1.5,2.0,5.0,10.0]...

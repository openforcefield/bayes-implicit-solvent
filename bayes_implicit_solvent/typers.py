from copy import deepcopy

import networkx as nx
import numpy as np
from scipy.stats import norm
from simtk import unit

from bayes_implicit_solvent.constants import RADIUS_UNIT
from bayes_implicit_solvent.smarts import atomic_number_dict
from bayes_implicit_solvent.utils import smarts_to_subsearch, convert_to_unitd_array


class DiscreteProposal():
    def sample(self, initial):
        raise (NotImplementedError())

    def log_prob_forward(self, initial, proposed):
        raise (NotImplementedError())

    def log_prob_reverse(self, initial, proposed):
        """Only one way to go backwards"""
        number_of_ways_to_go_backward = 1
        return - np.log(number_of_ways_to_go_backward)

    def log_prob_forward_over_reverse(self, initial, proposed):
        return self.log_prob_forward(initial, proposed) - self.log_prob_reverse(initial, proposed)

    def __call__(self, initial):
        return self.sample(initial)


class BondProposal(DiscreteProposal):
    def __init__(self, bondable_types):
        self.bondable_types = bondable_types
        print('bondable_types (~\'d onto atoms)\n\t', self.bondable_types)

    def log_prob_forward(self, initial_smirks, proposed_smirks=None):
        """In this case, have an equal chance of picking any possible proposed smirks"""
        closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']
        n_atoms = len(closing_bracket_indices)
        number_of_ways_to_go_forward = n_atoms * len(self.bondable_types)
        return - np.log(number_of_ways_to_go_forward)

    def sample(self, initial_smirks):
        """Find square brackets, and propose to any-bond it to any element of bondable_types

        [#1] --> [#1]~[#8]
        [#6] --> [#6]~[-1]

        # TODO: Also allow multiple bond specification
        # TODO: e.g. '[#7]-[#6]' --> '[#7]([#1])-[#6]'
        (Although in that example we could also specify this by (!H0) I guess
        """
        closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']

        n_atoms = len(closing_bracket_indices)

        if n_atoms == 0:
            raise (RuntimeError('no atoms found in this smirks pattern!\n{}'.format(initial_smirks)))
        ind_to_insert = np.random.choice(closing_bracket_indices) + 1

        decorator = '~{}'.format(np.random.choice(self.bondable_types))

        proposal_smirks = initial_smirks[:ind_to_insert] + decorator + initial_smirks[ind_to_insert:]

        return {
            'proposal': proposal_smirks,
            'log_prob_forward_over_reverse': self.log_prob_forward_over_reverse(initial_smirks, proposed_smirks),
        }


class AtomSpecificationProposal(DiscreteProposal):
    def __init__(self, atomic_specifiers):
        self.atomic_specifiers = atomic_specifiers
        print('atomic_specifiers (&\'d into atom definitions)\n\t', self.atomic_specifiers)

    def log_prob_forward(self, initial_smirks, proposed_smirks=None):
        """In this case, have an equal chance of picking any possible proposed smirks"""
        closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']
        n_atoms = len(closing_bracket_indices)
        number_of_ways_to_go_forward = n_atoms * len(self.atomic_specifiers)
        return - np.log(number_of_ways_to_go_forward)

    def sample(self, initial_smirks):
        """Look inside square brackets, and try &'ing the current definition with something"""

        closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']

        n_atoms = len(closing_bracket_indices)

        if n_atoms == 0:
            raise (RuntimeError('no atoms found in this smirks pattern!\n{}'.format(initial_smirks)))
        ind_to_insert = np.random.choice(closing_bracket_indices)

        decorator = '&{}'.format(np.random.choice(self.atomic_specifiers))

        proposal_smirks = initial_smirks[:ind_to_insert] + decorator + initial_smirks[ind_to_insert:]

        return {
            'proposal': proposal_smirks,
            'log_prob_forward_over_reverse': self.log_prob_forward_over_reverse(initial_smirks, proposal_smirks),
        }


class BondSpecificationProposal(DiscreteProposal):
    def __init__(self, bond_specifiers):
        self.bond_specifiers = bond_specifiers
        print('bond_specifiers (replace "~" bonds)\n\t', self.bond_specifiers)

    def log_prob_forward(self, initial, proposed):
        if '~' not in initial_smirks:
            # contained no any-bonds
            log_prob_forward = - np.inf
        else:
            n_bonds = len(initial_smirks.split('~')) - 1
            number_of_ways_to_go_forward = len(self.bond_specifiers) * n_bonds
            log_prob_forward = - np.log(number_of_ways_to_go_forward)
        return log_prob_forward

    def sample(self, initial_smirks):
        """Look at unspecified bonds, randomly replace them with one of the bond specifiers"""
        if '~' not in initial_smirks:
            # contained no any-bonds
            proposal_smirks = initial_smirks
            number_of_ways_to_go_forward = 0  # calling -np.log(0) clutters up the terminal a bit
            log_prob_forward = - np.inf
        else:
            components = initial_smirks.split('~')
            bonds = ['~'] * (len(components) - 1)
            bonds[np.random.randint(len(bonds))] = np.random.choice(self.bond_specifiers)

            # components[0] + bonds[0] + components[1] + bonds[1] + ... + components[-1]
            s = []
            for i in range(len(bonds)):
                s.append(components[i])
                s.append(bonds[i])
            s.append(components[-1])
            proposal_smirks = ''.join(s)

        return {
            'proposal': proposal_smirks,
            'log_prob_forward_over_reverse': self.log_prob_forward_over_reverse(initial_smirks, proposed_smirks),
        }


class SMIRKSElaborationProposal(DiscreteProposal):
    def __init__(self, smirks_elaborators):
        self.smirks_elaborators = smirks_elaborators
        self.log_prob_pick_elaborator = - np.log(len(self.smirks_elaborators))

    def log_prob_forward(self, initial, proposed):
        return self.log_prob_pick_elaborator

    def sample(self, initial_smirks):
        """Pick one of the smirks_elaborators at random, and sample its proposal"""
        proposal_dict = np.random.choice(self.smirks_elaborators).sample(initial_smirks)

        # print('smirks proposal: {} --> {}'.format(initial_smirks, proposal_dict['proposal']))
        proposal_dict['log_prob_forward_over_reverse'] += self.log_prob_forward(initial_smirks,
                                                                                proposal_dict['proposal'])
        return proposal_dict


from bayes_implicit_solvent.utils import cached_substructure_matches


class SMARTSTyper():
    def __init__(self, smarts_iter):
        """Base class for typing schemes that use SMARTS.

        Elaborations might structure the SMARTS collection as a list, tree, or other structure, and may
        assign types / parameters in different ways (e.g. on the label of the last node visited in depth-first search
        of the SMARTS tree, or in terms of contributions from all matches, or something along those lines).

        Parameters
        ----------
        smarts_iter : iterable of SMARTS strings

        Methods
        -------
        get_matches
        """

        self.smarts_iter = smarts_iter
        self.smarts_list = list(self.smarts_iter)  # flatten out the iterable
        self.subsearches = list(map(smarts_to_subsearch, self.smarts_list))

    def get_matches(self, oemol):
        """Get a binary matrix of which atoms are hit by which smarts

        Returns
        -------
        hits : binary array of shape (n_atoms, len(self.subsearches))
        """
        n_atoms = len(list(oemol.GetAtoms()))

        match_matrix = np.zeros((n_atoms, len(self.smarts_list)), dtype=bool)

        for i in range(len(self.subsearches)):
            match_matrix[:, i] = cached_substructure_matches(oemol, self.smarts_list[i])

        return match_matrix

    def __repr__(self):
        return 'SMARTSTyper with {} patterns: '.format(len(self.smarts_list)) + str(self.smarts_list)


class FlatGBTyper(SMARTSTyper):
    def __init__(self, smarts_list):
        """List of SMARTS patterns, last match wins

        Parameters
        ----------
        smarts_list : list of SMARTS strings
        """
        super().__init__(smarts_list)
        self.n_types = len(self.smarts_list)

    def get_indices_of_last_matches(self, match_matrix):
        """Process a binary match matrix into an integer array, assigning an integer to each atom."""
        assert (match_matrix.shape[1] == self.n_types)
        assert (np.all(match_matrix.sum(1) >= 1))  # every atom should be hit by at least one smarts pattern

        inds = np.arange(match_matrix.shape[1])[::-1]
        largest_match_inds = inds[np.argmax(match_matrix[:, ::-1], axis=1)]

        return largest_match_inds

    def get_gb_types(self, oemol):
        """For each atom in oemol, get the index of the last SMARTS string in the list that matches it"""

        match_matrix = self.get_matches(oemol)
        gb_types = self.get_indices_of_last_matches(match_matrix)
        return gb_types

    def __repr__(self):
        return 'GBTyper with {} types: '.format(self.n_types) + str(self.smarts_list)


class GBTypingTree():
    def __init__(self,
                 smirks_elaboration_proposal,
                 default_parameters={'radius': 0.1 * unit.nanometer},
                 proposal_sigma=0.01 * unit.nanometer,
                 un_delete_able_types=set(['*']),
                 max_nodes=100,
                 ):
        """We represent a typing scheme using a tree of elaborations (each child is an elaboration on its parent node)
        with types assigned using a last-match-wins scheme.

        We will impose some additional constraints on what are allowable typing trees in prior_checking.py, for example, we
        may require that the scheme not assign the root (wildcard) type to any atom in a base set of molecules (such as DrugBank).
        We may also require that the scheme not have any redundant types.
        """

        self.G = nx.DiGraph()
        self.default_parameters = default_parameters
        self.proposal_sigma = proposal_sigma
        self.un_delete_able_types = un_delete_able_types
        self.max_nodes = max_nodes
        # initialize wildcard root node
        self.G.add_node('*', **self.default_parameters)
        self.smirks_elaboration_proposal = smirks_elaboration_proposal
        self.update_node_order()

    def update_node_order(self):
        """Order the nodes by a breadth-first search"""
        self.ordered_nodes = list(nx.breadth_first_search.bfs_tree(self.G, '*').nodes())

    def apply_to_molecule(self, molecule):
        """Assign types based on last-match-wins during breadth-first search

        Return integer array of types
        """
        # TODO: For tall trees, will be more efficient to only call oechem on the nodes that are actually
        # TODO: visited during breadth-first search, rather than calling oechem on all of the nodes first
        return FlatGBTyper(self.ordered_nodes).get_gb_types(molecule)

    def assign_radii(self, molecule):
        """Return a unit'd array of radii"""
        types = self.apply_to_molecule(molecule)
        radii = [self.get_radius(self.ordered_nodes[t]) for t in types]
        return convert_to_unitd_array(radii)

    def apply_to_molecule_list(self, molecules):
        """Assign types to all molecules in a list"""
        return list(map(self.apply_to_molecule, molecules))

    def is_leaf(self, node):
        """Return True if node has no descendents"""
        return len(self.G[node]) == 0

    def is_delete_able(self, node):
        """Return True if node can be deleted.

        Currently, we prevent deletion of:
        * non-leaf nodes
        * the wild-card node
        * any of the atomic-number nodes"""
        # TODO: refactor this a bit
        return self.is_leaf(node) and (not (node in self.un_delete_able_types))

    def is_decorate_able(self, node):
        """Return True if we can create direct descendant types from this node.

        Currently, we prevent proposing new direct descendants of the root '*'
        """
        return node != '*'

    @property
    def nodes(self):
        return list(self.G.nodes())

    @property
    def number_of_nodes(self):
        """The total number of nodes in the graph"""
        return len(self.nodes)

    @property
    def leaves(self):
        return [n for n in self.nodes if self.is_leaf(n)]

    @property
    def number_of_leaves(self):
        """The total number of leaf nodes in the graph"""
        return len(self.leaves)

    @property
    def delete_able_nodes(self):
        return [n for n in self.nodes if self.is_delete_able(n)]

    @property
    def number_of_delete_able_nodes(self):
        """The total number of delete-able nodes in the graph"""
        return len(self.delete_able_nodes)

    @property
    def decorate_able_nodes(self):
        return [n for n in self.nodes if self.is_decorate_able(n)]

    @property
    def number_of_decorate_able_nodes(self):
        """The total number of decorate-able nodes in the graph"""
        return len(self.decorate_able_nodes)

    def sample_node_uniformly_at_random(self):
        """Select any node, including the root"""
        if self.number_of_nodes == 0:
            raise (RuntimeError('No nodes left!'))

        return np.random.choice(self.nodes)

    def sample_delete_able_node_uniformly_at_random(self):
        """Select among the delete-able leaf nodes"""
        if self.number_of_delete_able_nodes == 0:
            raise (RuntimeError('No delete-able nodes left!'))

        return np.random.choice(self.delete_able_nodes)

    def sample_leaf_node_uniformly_at_random(self):
        """Select any node that has no descendents"""
        if self.number_of_leaves == 0:
            raise (RuntimeError('No leaf nodes left!'))

        return np.random.choice(self.leaves)

    def sample_decorate_able_node_uniformly_at_random(self):
        """Select any node we can propose to decorate"""
        if self.number_of_decorate_able_nodes == 0:
            raise (RuntimeError('No decorate-able nodes left!'))

        return np.random.choice(self.decorate_able_nodes)

    def add_child(self, child_smirks, parent_smirks):
        """Create a new type, and add a directed edge from parent to child"""
        self.G.add_node(child_smirks, **self.default_parameters)
        self.G.add_edge(parent_smirks, child_smirks)
        self.update_node_order()

    def get_parent_type(self, smirks):
        """Get the parent of a given type.
        If we query the root, return None.
        If there is more than one parent type, raise an error"""
        incoming_edges = list(self.G.in_edges(smirks))
        if len(incoming_edges) == 0:
            return None
        elif len(incoming_edges) > 1:
            raise (RuntimeError('More than one parent type!'))
        else:
            return incoming_edges[0][0]

    def remove_node(self, smirks, only_allow_leaf_deletion=True):
        """Remove a node from the graph, optionally prevent deletion of interior nodes"""
        if only_allow_leaf_deletion:
            if self.is_leaf(smirks):
                self.G.remove_node(smirks)
            else:
                raise (RuntimeError('Attempted to delete a non-leaf node!'))
        else:
            self.G.remove_node(smirks)

        self.update_node_order()

    def get_radius(self, smirks):
        """Get the value of the "radius" property for the smirks type"""
        return nx.get_node_attributes(self.G, 'radius')[smirks]

    def get_radii(self):
        """Get the "radius" properties for all nodes as an array"""
        radii = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes):
            radii[i] = self.get_radius(self.ordered_nodes[i]) / RADIUS_UNIT
        return radii

    def set_radius(self, smirks, radius):
        """Set the value of the "radius" property for the smirks type"""
        nx.set_node_attributes(self.G, {smirks: radius}, name='radius')

    def sample_creation_proposal(self):
        """Propose to randomly decorate an existing type to create a new type,
        with a slightly different radius than the existing type.
        Return a dict containing the new typing tree and the ratio of forward and reverse proposal probabilities."""

        # Create a copy of the current typer, which we will modify and return
        proposal = deepcopy(self)

        # sample a parent node uniformly at random from decorate-able nodes
        parent_smirks = self.sample_decorate_able_node_uniformly_at_random()

        # create a new type by elaborating on the parent type
        elaboration_proposal_dict = self.smirks_elaboration_proposal.sample(parent_smirks)
        child_smirks = elaboration_proposal_dict['proposal']

        # create a new type by elaborating on the parent type
        proposal.add_child(child_smirks=child_smirks, parent_smirks=parent_smirks)

        # set the radius of the new type as a gaussian perturbation of the parent's radius
        # TODO: this bit hard-codes that the only parameter we're interested in is 'radius'
        # TODO: Also interested in scale, for example, and potentially also the SASA term
        parent_radius = self.get_radius(parent_smirks)
        proposal_radius = self.proposal_sigma * np.random.randn() + parent_radius
        proposal.set_radius(child_smirks, proposal_radius)

        # compute the log ratio of forward and reverse proposal probabilities
        # TODO: accumulate these properties in a less manual / error-prone way
        delta_radius = proposal_radius - parent_radius
        delta = delta_radius / RADIUS_UNIT
        sigma = self.proposal_sigma / RADIUS_UNIT

        log_prob_forward = - np.log(self.number_of_decorate_able_nodes) \
                           + norm.logpdf(delta, loc=0, scale=sigma)
        log_prob_reverse = - np.log(proposal.number_of_delete_able_nodes)
        log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse
        # include contributions from all the choices made in the smirks_elaboration_proposal
        log_prob_forward_over_reverse += elaboration_proposal_dict['log_prob_forward_over_reverse']

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
                }

    def sample_deletion_proposal(self):
        """Sample a (delete-able) leaf node randomly and propose to delete it.
        Return a dict containing the new typing tree and the ratio of forward and reverse proposal probabilities."""

        # Create a copy of the current typer, which we will modify and return
        proposal = deepcopy(self)

        # Delete a delete-able node leaf at random
        leaf_to_delete = self.sample_delete_able_node_uniformly_at_random()
        proposal.remove_node(leaf_to_delete)

        # compute the log ratio of forward and reverse proposal probabilities

        # the forward probability is given by just how many delete-able nodes there are
        # prob_forward = 1 / self.number_of_delete_able_nodes
        log_prob_forward = - np.log(self.number_of_delete_able_nodes)

        # the reverse probability is how likely we are to have grown the leaf we just deleted
        # that's (probability of picking the parent) * (probability of picking that decorator given that we picked the parent)
        # * (probability of picking that particular radius given that we picked the parent)
        parent = self.get_parent_type(leaf_to_delete)
        leaf_radius = self.get_radius(leaf_to_delete)
        parent_radius = self.get_radius(parent)
        delta_radius = leaf_radius - parent_radius
        delta = delta_radius / RADIUS_UNIT
        sigma = self.proposal_sigma / RADIUS_UNIT

        log_prob_reverse = norm.logpdf(delta, loc=0, scale=sigma)
        log_prob_reverse += - np.log(proposal.number_of_decorate_able_nodes)
        log_prob_reverse += proposal.smirks_elaboration_proposal.log_prob_forward(parent, None)
        log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
                }

    @property
    def probability_of_sampling_a_create_proposal(self):
        """Probability of attempting to create a new type"""

        N, max_N = self.number_of_nodes, self.max_nodes

        # if we already have the maximum allowable number of nodes, cannot create another
        if N >= max_N:
            if N > max_N:
                raise (RuntimeError('Tree contains {} nodes, exceeding max_nodes={}'.format(N, max_N)))
            return 0.0

        # if we have no delete-able nodes, must sample create proposal
        elif self.number_of_delete_able_nodes == 0:
            return 1.0

        # otherwise, flip a coin
        else:
            return 0.5

    @property
    def probability_of_sampling_a_delete_proposal(self):
        """Probability of attempting to delete an existing type"""
        return 1.0 - self.probability_of_sampling_a_create_proposal

    def sample_create_delete_proposal(self):
        """Randomly propose to create or delete a type"""

        move_sampler_dict = {'create': lambda x: x.sample_creation_proposal(),
                             'delete': lambda x: x.sample_deletion_proposal()}
        move_prob_dict = {'create': lambda x: x.probability_of_sampling_a_create_proposal,
                          'delete': lambda x: x.probability_of_sampling_a_delete_proposal,
                          }

        assert (len(move_sampler_dict) == 2)

        move, reverse_move = ('create', 'delete')

        p_create = self.probability_of_sampling_a_create_proposal
        if np.random.rand() <= p_create:
            p_forward = p_create
        else:
            p_forward = 1.0 - p_create
            move, reverse_move = (reverse_move, move)

        assert (p_forward > 0)

        proposal_dict = move_sampler_dict[move](self)
        p_reverse = move_prob_dict[reverse_move](proposal_dict['proposal'])
        log_prob_forward_over_reverse_correction = np.log(p_forward / p_reverse)

        assert (np.isfinite(log_prob_forward_over_reverse_correction))

        proposal_dict['log_prob_forward_over_reverse'] += log_prob_forward_over_reverse_correction

        return proposal_dict

    def sample_radius_perturbation_proposal(self):
        """Pick a type at random and propose to perturb its radius slightly"""

        proposal = deepcopy(self)

        node = self.sample_node_uniformly_at_random()
        initial_radius = self.get_radius(node)
        proposal_radius = self.proposal_sigma * np.random.randn() + initial_radius
        proposal.set_radius(node, proposal_radius)

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': 0,  # symmetric
                }

    def __repr__(self):
        """Format tree nicely"""
        depth_dict = nx.shortest_path_length(self.G, source='*')
        prefix = '|-'
        lines = []
        radii = []

        for n in nx.depth_first_search.dfs_preorder_nodes(self.G, '*'):
            if depth_dict[n] > 0:
                lines.append('  ' * (depth_dict[n] - 1) + prefix + n)
            else:
                lines.append(n)
            radii.append('(r = {:.5} nm)'.format(str(self.get_radius(n) / RADIUS_UNIT)))

        max_length = max(np.array(list(map(len, lines))) + np.array(list(map(len, radii))))
        width = max_length + 4

        return '\n'.join([lines[i] + radii[i].rjust(width - len(lines[i])) for i in range(len(lines))])


if __name__ == '__main__':
    initial_smirks = '[#8]'

    print('using the following decorators:')

    all_bond_specifiers = ['@', '-', '#', '=', ':']

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

    for _ in range(50):
        proposal = smirks_elaboration_proposal(initial_smirks)
        proposed_smirks = proposal['proposal']
        print('proposal: {} --> {}'.format(initial_smirks, proposed_smirks))
        lpfr = proposal['log_prob_forward_over_reverse']
        print('log prob forward over reverse = {}'.format(lpfr))
        if np.random.rand() < np.exp(-lpfr):
            initial_smirks = proposed_smirks

    np.random.seed(0)
    initial_tree = GBTypingTree(smirks_elaboration_proposal=smirks_elaboration_proposal)
    for base_type in atomic_number_dict.keys():
        initial_tree.add_child(child_smirks=base_type, parent_smirks='*')
    print(initial_tree)
    creation_proposals = [initial_tree.sample_creation_proposal()]

    for _ in range(10):
        print('proposal:')
        print(creation_proposals[-1]['proposal'])
        print('log_prob_forward_over_reverse')
        print(creation_proposals[-1]['log_prob_forward_over_reverse'])

        creation_proposals.append(creation_proposals[-1]['proposal'].sample_creation_proposal())

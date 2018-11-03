from copy import deepcopy

import networkx as nx
import numpy as np
from openeye import oechem
from scipy.stats import norm
from simtk import unit

from bayes_implicit_solvent.smarts import atomic_number_dict
from bayes_implicit_solvent.utils import smarts_to_subsearch

print('using the following decorators:')

bond_specifiers = ['@', '-', '#', '=', ':']
print('bond_specifiers (replace "~" bonds)\n\t', bond_specifiers)

bondable_types = list(atomic_number_dict.keys())

# this is a group that appears a 9 times in the smirnoff99frosst nonbonded definitions for hydrogens
bondable_types.append('[(#7,#8,#9,#16,#17,#35)]')

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
atomic_specifiers = list(chain(*all_specifier_lists))

print('atomic_specifiers (&\'d into atom definitions)\n\t', atomic_specifiers)

bondable_types += ['[{}]'.format(s) for s in atomic_specifiers]

print('bondable_types (~\'d onto atoms)\n\t', bondable_types)

all_decorators = bondable_types + atomic_specifiers + bond_specifiers

def sample_bond_proposal(initial_smirks):
    """Find square brackets, and propose to any-bond it to any element of bondable_types

    [#1] --> [#1]~[#8]
    [#1] --> [#6]~[-1]

    # TODO: Also allow multiple bond specification
    # TODO: e.g. '[#7]-[#6]' --> '[#7]([#1])-[#6]'
    (Although in that example we could also specify this by (!H0) I guess
    """
    closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']

    n_atoms = len(closing_bracket_indices)

    if n_atoms == 0:
        raise(RuntimeError('no atoms found in this smirks pattern!\n{}'.format(initial_smirks)))
    ind_to_insert = np.random.choice(closing_bracket_indices) + 1

    decorator = '~{}'.format(np.random.choice(bondable_types))

    proposal_smirks = initial_smirks[:ind_to_insert] + decorator + initial_smirks[ind_to_insert:]

    number_of_ways_to_go_forward = n_atoms * len(bondable_types)
    number_of_ways_to_go_backward = 1

    log_prob_forward = - np.log(number_of_ways_to_go_forward)
    log_prob_reverse = - np.log(number_of_ways_to_go_backward)

    log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse
    return {
        'proposal': proposal_smirks,
        'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
    }

def sample_atom_specification_proposal(initial_smirks):
    """Look inside square brackets, and try &'ing the current definition with something"""

    closing_bracket_indices = [i for i in range(len(initial_smirks)) if initial_smirks[i] == ']']

    n_atoms = len(closing_bracket_indices)

    if n_atoms == 0:
        raise(RuntimeError('no atoms found in this smirks pattern!\n{}'.format(initial_smirks)))
    ind_to_insert = np.random.choice(closing_bracket_indices)

    decorator = '&{}'.format(np.random.choice(atomic_specifiers))

    proposal_smirks = initial_smirks[:ind_to_insert] + decorator + initial_smirks[ind_to_insert:]

    number_of_ways_to_go_forward = n_atoms * len(atomic_specifiers)
    number_of_ways_to_go_backward = 1

    log_prob_forward = - np.log(number_of_ways_to_go_forward)
    log_prob_reverse = - np.log(number_of_ways_to_go_backward)

    log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse
    return {
        'proposal': proposal_smirks,
        'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
    }

def sample_bond_specification_proposal(initial_smirks):
    """Look at unspecified bonds, randomly replace them with one of the bond specifiers"""
    if '~' not in initial_smirks:
        # contained no any-bonds
        proposal_smirks = initial_smirks
        number_of_ways_to_go_forward = 0 # calling -np.log(0) clutters up the terminal a bit
        log_prob_forward = - np.inf
    else:
        components = initial_smirks.split('~')
        bonds = ['~'] * (len(components) - 1)
        bonds[np.random.randint(len(bonds))] = np.random.choice(bond_specifiers)

        # components[0] + bonds[0] + components[1] + bonds[1] + ... + components[-1]
        s = []
        for i in range(len(bonds)):
            s.append(components[i])
            s.append(bonds[i])
        s.append(components[-1])
        proposal_smirks = ''.join(s)
        number_of_ways_to_go_forward = len(bond_specifiers) * len(bonds)
        log_prob_forward = - np.log(number_of_ways_to_go_forward)

    number_of_ways_to_go_backward = 1


    log_prob_reverse = - np.log(number_of_ways_to_go_backward)

    log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse
    return {
        'proposal': proposal_smirks,
        'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
    }

smirks_elaborators = [
    sample_bond_proposal,
    sample_atom_specification_proposal,
    sample_bond_specification_proposal,
]

def sample_smirks_elaboration_proposal(initial_smirks):
    """Pick one of the smirks_elaborators at random, and sample its proposal"""
    log_prob_pick_elaborator = - np.log(len(smirks_elaborators))
    proposal_dict = np.random.choice(smirks_elaborators)(initial_smirks)
    #print('smirks proposal: {} --> {}'.format(initial_smirks, proposal_dict['proposal']))
    proposal_dict['log_prob_forward_over_reverse'] += log_prob_pick_elaborator
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
                 default_parameters={'radius': 0.1 * unit.nanometer},
                 proposal_sigma=0.01 * unit.nanometer,
                 un_delete_able_types=set(['*']),
                 sample_smirks_elaboration_proposal=sample_smirks_elaboration_proposal,
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
        # initialize wildcard root node
        self.G.add_node('*', **self.default_parameters)
        self.sample_smirks_elaboration_proposal = sample_smirks_elaboration_proposal
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
        return np.array([r / r.unit for r in radii]) * radii[0].unit

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
    def number_of_nodes(self):
        """The total number of nodes in the graph"""
        return len(self.G.nodes())

    @property
    def number_of_leaves(self):
        """The total number of leaf nodes in the graph"""
        return sum(list(map(lambda n: self.is_leaf(n), self.G.nodes())))

    @property
    def number_of_delete_able_nodes(self):
        """The total number of delete-able nodes in the graph"""
        return sum(list(map(lambda n: self.is_delete_able(n), self.G.nodes())))

    @property
    def number_of_decorate_able_nodes(self):
        """The total number of decorate-able nodes in the graph"""
        return sum(list(map(lambda n: self.is_decorate_able(n), self.G.nodes())))

    def sample_node_uniformly_at_random(self):
        """Select any node, including the root"""
        nodes = list(self.G.nodes())
        if len(nodes) == 0:
            raise (RuntimeError('No nodes left!'))
        return nodes[np.random.randint(len(nodes))]

    def sample_delete_able_node_uniformly_at_random(self):
        """Select among the delete-able leaf nodes"""
        delete_able_nodes = [n for n in self.G.nodes() if self.is_delete_able(n)]
        if len(delete_able_nodes) == 0:
            raise (RuntimeError('No delete-able nodes left!'))
        return delete_able_nodes[np.random.randint(len(delete_able_nodes))]

    def sample_leaf_node_uniformly_at_random(self):
        """Select any node that has no descendents"""
        leaf_nodes = [n for n in self.G.nodes() if self.is_leaf(n)]
        if len(leaf_nodes) == 0:
            raise (RuntimeError('No leaf nodes left!'))
        return leaf_nodes[np.random.randint(len(leaf_nodes))]

    def sample_decorate_able_node_uniformly_at_random(self):
        """Select any node we can propose to decorate"""
        decorate_able_nodes = [n for n in self.G.nodes() if self.is_decorate_able(n)]
        if len(decorate_able_nodes) == 0:
            raise (RuntimeError('No decorate-able nodes left!'))
        return decorate_able_nodes[np.random.randint(len(decorate_able_nodes))]

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
        elaboration_proposal_dict = self.sample_smirks_elaboration_proposal(parent_smirks)
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
        delta = delta_radius / delta_radius.unit
        sigma = self.proposal_sigma / delta_radius.unit


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
        log_prob_forward = - np.log(self.number_of_delete_able_nodes)
        parent = self.get_parent_type(leaf_to_delete)
        leaf_radius = self.get_radius(leaf_to_delete)
        parent_radius = self.get_radius(parent)
        delta_radius = leaf_radius - parent_radius
        delta = delta_radius / delta_radius.unit
        sigma = self.proposal_sigma / delta_radius.unit

        # TODO: the "- np.log(len(all_decorators))" term is almost certainly incorrect!
        # (I think we need to sum up over all of the smirks nodes, which might have more than
        # one atom or bond. For initial tinkering, this will be close-ish, but it's almost certainly
        # wrong!)
        log_prob_reverse = - np.log(proposal.number_of_decorate_able_nodes) \
                           - np.log(len(all_decorators)) \
                           + norm.logpdf(delta, loc=0, scale=sigma)
        log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
                }

    def sample_create_delete_proposal(self):
        """Flip a coin and either propose to create or delete a type"""

        if self.number_of_delete_able_nodes == 0:
            return self.sample_creation_proposal()
        # TODO: Double-check, probably need to modify log_prob_forward_over_reverse at this boundary?

        if np.random.rand() <= 0.5:
            return self.sample_creation_proposal()
        else:
            return self.sample_deletion_proposal()

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
            radii.append('(r = {:.5} nm)'.format(str(self.get_radius(n) / unit.nanometer)))

        max_length = max(np.array(list(map(len, lines))) + np.array(list(map(len, radii))))
        width = max_length + 4

        return '\n'.join([lines[i] + radii[i].rjust(width - len(lines[i])) for i in range(len(lines))])


if __name__ == '__main__':
    initial_smirks = '[#8]'
    for _ in range(50):
        proposal = sample_smirks_elaboration_proposal(initial_smirks)
        proposed_smirks = proposal['proposal']
        print('proposal: {} --> {}'.format(initial_smirks, proposed_smirks))
        lpfr = proposal['log_prob_forward_over_reverse']
        print('log prob forward over reverse = {}'.format(lpfr))
        if np.random.rand() < np.exp(-lpfr):
            initial_smirks = proposed_smirks

    np.random.seed(0)
    initial_tree = GBTypingTree()
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

from copy import deepcopy

import networkx as nx
import numpy as np
from openeye import oechem
from scipy.stats import norm
from simtk import unit

from bayes_implicit_solvent.smarts import atomic_number_dict
#from bayes_implicit_solvent.smarts import decorators as decorator_dict
from bayes_implicit_solvent.utils import smarts_to_subsearch

decorators = ['~{}'.format(a) for a in atomic_number_dict]
print('using the following decorators')

def sample_decorator_uniformly_at_random():
    return decorators[np.random.randint(len(decorators))]


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
        mol_ = oechem.OEMol(oemol)
        n_atoms = len(list(mol_.GetAtoms()))

        match_matrix = np.zeros((n_atoms, len(self.subsearches)), dtype=bool)

        for i in range(len(self.subsearches)):
            for match in self.subsearches[i].Match(mol_, False):
                match_atoms = match.GetTargetAtoms()
                for a in match_atoms:
                    match_matrix[a.GetIdx(), i] = True
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
    def __init__(self, default_parameters={'radius': 0.1 * unit.nanometer}, proposal_sigma=0.01 * unit.nanometer):
        """We represent a typing scheme using a tree of elaborations (each child is an elaboration on its parent node)
        with types assigned using a last-match-wins scheme.

        We will impose some additional constraints on what are allowable typing trees in prior_checking.py, for example, we
        may require that the scheme not assign the root (wildcard) type to any atom in a base set of molecules (such as DrugBank).
        We may also require that the scheme not have any redundant types.
        """

        self.G = nx.DiGraph()
        self.default_parameters = default_parameters
        self.proposal_sigma = proposal_sigma

        # initialize wildcard root node
        self.G.add_node('*', **self.default_parameters)

    @property
    def ordered_nodes(self):
        """Order the nodes by a breadth-first search"""
        return list(nx.breadth_first_search.bfs_tree(self.G, '*').nodes())

    def apply_to_molecule(self, molecule):
        """Assign types based on last-match-wins during breadth-first search

        Return integer array of types
        """
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

    @property
    def number_of_nodes(self):
        """Return the total number of nodes in the graph"""
        return len(self.G.nodes())

    @property
    def number_of_leaves(self):
        """Return the total number of leaf nodes in the graph"""
        return sum(list(map(lambda n: self.is_leaf(n), self.G.nodes())))

    def sample_node_uniformly_at_random(self):
        """Select any node, including the root"""
        nodes = list(self.G.nodes())
        if len(nodes) == 0:
            raise (RuntimeError('No nodes left!'))
        return nodes[np.random.randint(len(nodes))]

    def sample_leaf_node_uniformly_at_random(self):
        """Select any node that has no descendents"""
        leaf_nodes = [n for n in self.G.nodes() if self.is_leaf(n)]
        if len(leaf_nodes) == 0:
            raise (RuntimeError('No leaf nodes left!'))
        return leaf_nodes[np.random.randint(len(leaf_nodes))]

    def add_child(self, child_smirks, parent_smirks):
        """Create a new type, and add a directed edge from parent to child"""
        self.G.add_node(child_smirks, **self.default_parameters)
        self.G.add_edge(parent_smirks, child_smirks)

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

        # sample a parent node uniformly at random
        parent_smirks = self.sample_node_uniformly_at_random()

        # sample a decorator uniformly at random
        decorator = sample_decorator_uniformly_at_random()

        # create a new type by decorating the parent type
        child_smirks = parent_smirks + decorator
        proposal.add_child(child_smirks=child_smirks, parent_smirks=parent_smirks)

        # set the radius of the new type as a gaussian perturbation of the parent's radius
        # TODO: this bit hard-codes that the only parameter we're interested in is 'radius'
        parent_radius = self.get_radius(parent_smirks)
        proposal_radius = self.proposal_sigma * np.random.randn() + parent_radius
        proposal.set_radius(child_smirks, proposal_radius)

        # compute the log ratio of forward and reverse proposal probabilities
        # TODO: accumulate these properties in a less manual / error-prone way
        delta_radius = proposal_radius - parent_radius
        delta = delta_radius / delta_radius.unit
        sigma = self.proposal_sigma / delta_radius.unit
        log_prob_forward = - np.log(len(self.G.nodes())) - np.log(len(decorators)) + norm.logpdf(delta, loc=0,
                                                                                                 scale=sigma)
        log_prob_reverse = - np.log(proposal.number_of_leaves)
        log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
                }

    def sample_deletion_proposal(self):
        """Sample a leaf node randomly and propose to delete it.
        Return a dict containing the new typing tree and the ratio of forward and reverse proposal probabilities."""

        # Create a copy of the current typer, which we will modify and return
        proposal = deepcopy(self)

        # Delete a leaf at random
        leaf_to_delete = self.sample_leaf_node_uniformly_at_random()
        proposal.remove_node(leaf_to_delete)

        # compute the log ratio of forward and reverse proposal probabilities
        log_prob_forward = - np.log(self.number_of_leaves)
        parent = self.get_parent_type(leaf_to_delete)
        leaf_radius = self.get_radius(leaf_to_delete)
        parent_radius = self.get_radius(parent)
        delta_radius = leaf_radius - parent_radius
        delta = delta_radius / delta_radius.unit
        sigma = self.proposal_sigma / delta_radius.unit
        # TODO: Double-check that the 1/N_decorator bit is okay
        log_prob_reverse = - np.log(proposal.number_of_nodes) - np.log(len(decorators)) + norm.logpdf(delta, loc=0,
                                                                                                      scale=sigma)
        log_prob_forward_over_reverse = log_prob_forward - log_prob_reverse

        return {'proposal': proposal,
                'log_prob_forward_over_reverse': log_prob_forward_over_reverse,
                }

    def sample_create_delete_proposal(self):
        """Flip a coin and either propose to create or delete a type."""
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
                'log_prob_forward_over_reverse': 0, # symmetric
                }

    def __repr__(self):
        # TODO: Less-janky text representation
        s = 'parent --> child\n'
        for e in self.G.edges():
            s += '\t' + e[0] + ' --> ' + e[1] + '\n'
        return s


if __name__ == '__main__':
    np.random.seed(0)
    initial_tree = GBTypingTree()
    for base_type in atomic_number_dict.keys():
        initial_tree.add_child(child_smirks=base_type, parent_smirks='*')
    print(initial_tree)
    creation_proposals = [initial_tree.sample_creation_proposal()]

    for _ in range(10):
        print('proposal:')
        print(creation_proposals[-1]['proposal'])
        print('log_p_reverse_over_forward')
        print(creation_proposals[-1]['log_prob_reverse_over_forward'])

        creation_proposals.append(creation_proposals[-1]['proposal'].sample_creation_proposal())

import numpy as np
from openeye import oechem
from bayes_implicit_solvent.utils import smarts_to_subsearch
from bayes_implicit_solvent.smarts import decorators as decorator_dict

decorators = sorted(list(decorator_dict.keys()))
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
        self.smarts_list = list(self.smarts_iter) # flatten out the iterable
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

if __name__ == '__main__':
    pass
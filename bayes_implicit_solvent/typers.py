import numpy as np
from openeye import oechem
from bayes_implicit_solvent.utils import smarts_to_subsearch


class GBTyper():
    def __init__(self, smarts_list):
        """List of SMARTS patterns, last match wins

        Parameters
        ----------
        smarts_list : list of SMARTS strings
        """
        self.smarts_list = smarts_list
        self.subsearches = list(map(smarts_to_subsearch, self.smarts_list))
        self.n_types = len(self.smarts_list)

    def get_matches(self, oemol):
        """Get a binary matrix of which atoms are hit by which smarts

        Returns
        -------
        hits : binary array of shape (n_atoms, self.n_types)
        """
        mol_ = oechem.OEMol(oemol)
        n_atoms = len(list(mol_.GetAtoms()))

        match_matrix = np.zeros((n_atoms, self.n_types), dtype=bool)

        for i in range(self.n_types):
            for match in self.subsearches[i].Match(mol_, False):
                match_atoms = match.GetTargetAtoms()
                for a in match_atoms:
                    match_matrix[a.GetIdx(), i] = True
        return match_matrix

    def get_indices_of_last_matches(self, match_matrix):
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

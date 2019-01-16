import numpy as np

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.solvation_free_energy import smiles_list, db

# TODO: Refactor duplicated test setup to use pytest fixtures
def test_molecule_gets_right_smiles():
    for _ in range(100):
        smiles = smiles_list[np.random.randint(len(smiles_list))]
        mol = Molecule(smiles, vacuum_samples=[])
        assert (smiles == mol.smiles)


def test_molecule_consistent_with_smiles_list():
    for _ in range(100):
        smiles = smiles_list[np.random.randint(len(smiles_list))]
        mol = Molecule(smiles, vacuum_samples=[])

        assert (mol.smiles == smiles_list[mol.mol_index_in_smiles_list])


def test_molecule_consistent_with_db():
    for _ in range(100):
        smiles = smiles_list[np.random.randint(len(smiles_list))]
        mol = Molecule(smiles, vacuum_samples=[])
        assert (mol.smiles == db[mol.mol_index_in_freesolv][1])

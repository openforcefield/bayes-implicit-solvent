import numpy as np
import pytest

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.solvation_free_energy import smiles_list, db


@pytest.fixture
def initialize_molecule_list():
    mol_smiles_pairs = []
    for _ in range(100):
        smiles = smiles_list[np.random.randint(len(smiles_list))]
        mol = Molecule(smiles, vacuum_samples=[])
        mol_smiles_pairs.append((mol, smiles))
    return mol_smiles_pairs


def test_molecule_gets_right_smiles(initialize_molecule_list):
    for (mol, smiles) in initialize_molecule_list:
        assert (smiles == mol.smiles)


def test_molecule_consistent_with_smiles_list(initialize_molecule_list):
    for (mol, smiles) in initialize_molecule_list:
        assert (mol.smiles == smiles_list[mol.mol_index_in_smiles_list])


def test_molecule_consistent_with_db(initialize_molecule_list):
    for (mol, smiles) in initialize_molecule_list:
        assert (mol.smiles == db[mol.mol_index_in_freesolv][1])

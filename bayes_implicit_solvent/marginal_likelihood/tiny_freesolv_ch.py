"""7 compounds containing only carbon and hydrogen,
and having only two topological symmetry classes each

ethane
benzene
cyclopentane
ethylene
methane
cyclopropane
cyclohexane
"""

import numpy as np
from openeye.oechem import OEPerceiveSymmetry
from simtk import unit

from bayes_implicit_solvent.freesolv import cid_to_smiles
from bayes_implicit_solvent.molecule import Molecule


def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = np.load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj


from glob import glob
from pkg_resources import resource_filename

ll = 'gaussian'
n_conf = 2

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))

mols = []

n_configuration_samples = n_conf

for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning], ll=ll)
    OEPerceiveSymmetry(mol.mol)
    atoms = list(mol.mol.GetAtoms())
    symmetry_types = np.array([atom.GetSymmetryClass() for atom in atoms])

    if (len(set(symmetry_types)) == 2) and (set([a.element.symbol for a in mol.top.atoms()]) == {'C', 'H'}):
        print(mol.mol_name)
        mols.append(mol)

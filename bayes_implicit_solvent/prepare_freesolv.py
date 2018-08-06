# prepares SMIRNOFF'd openmm systems ready to simulate
# (constructs OEMOL objects, assigns partial charges, etc.)

# takes approximately 3hours to run serially, and outputs a pickle file

import os

from pkg_resources import resource_filename
path_to_freesolv = resource_filename('bayes_implicit_solvent', 'data/FreeSolv-0.51/database.txt')

from tqdm import tqdm

with open(path_to_freesolv, 'r') as f:
    freesolv = f.read()

legend = freesolv.split('\n')[2].split('; ')
print(list(zip(range(len(legend)), legend)))

db = []
for entry in freesolv.split('\n')[3:-1]:
    db.append(entry.split('; '))

smiles_list = [entry[1] for entry in db]

import numpy as np

from openeye import oeomega  # Omega toolkit

omega = oeomega.OEOmega()
omega.SetMaxConfs(800)  # Best-practice? Expensive...
omega.SetIncludeInput(False)
omega.SetStrictStereo(True)  # Refuse to generate conformers if stereochemistry not provided

from openforcefield.typing.engines.smirnoff import ForceField, generateTopologyFromOEMol
from openeye import oechem  # OpenEye Python toolkits

from openeye import oequacpac  # Charge toolkit

ff = ForceField(os.path.join(data_path, 'smirnoff99Frosst.offxml'))


def generate_oemol(smiles):
    """Add hydrogens, assign charges, generate conformers, and return molecule"""
    mol = oechem.OEMol()
    chargeEngine = oequacpac.OEAM1BCCCharges()
    oechem.OEParseSmiles(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
    status = omega(mol)
    if not status: print("Something went wrong in `generate_oemol({})!".format(smiles))
    oechem.OETriposAtomNames(mol)
    oequacpac.OESetNeutralpHModel(mol)
    oequacpac.OEAssignCharges(mol, chargeEngine)
    _ = generateTopologyFromOEMol(mol)
    return mol


def generate_mol_top_sys_pos(smiles):
    """Generate an openmm topology, openmm system, and coordinate array from a smiles string"""
    print(smiles)
    mol = generate_oemol(smiles)

    coord_dict = mol.GetCoords()
    positions = np.array([coord_dict[key] for key in coord_dict])

    topology = generateTopologyFromOEMol(mol)

    system = ff.createSystem(topology, [mol])

    return mol, topology, system, positions


sorted_smiles = sorted(list(set(smiles_list)), key=len)[::-1]

mol_top_sys_pos_list = []

for smiles in tqdm(sorted_smiles):
    mol_top_sys_pos_list.append(tuple(generate_mol_top_sys_pos(smiles)))

from pickle import dump

with open(resource_filename('bayes_implicit_solvent', 'data/mol_top_sys_pos.pkl'), 'wb') as f:
    dump(mol_top_sys_pos_list, f)

with open(resource_filename('bayes_implicit_solvent', 'data/sorted_smiles.pkl'), 'wb') as f:
    dump(sorted_smiles, f)

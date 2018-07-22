# For each SMILES string in FreeSolv, parameterizes the molecule using SMIRNOFF and dumps topology, system, positions

import numpy as np
from multiprocessing import Pool
n_subprocesses = 32

def load_freesolv(path="../FreeSolv-0.51/database.txt"):
    """Loads the freesolv database as a list of lists of strings"""
    with open(path, 'r') as f:
        freesolv = f.read()

    legend = freesolv.split('\n')[2].split('; ')
    db = []
    for entry in freesolv.split('\n')[3:-1]:
        db.append(entry.split('; '))
    return legend, db


legend, db = load_freesolv()
from openeye import oeomega  # Omega toolkit

omega = oeomega.OEOmega()
omega.SetMaxConfs(800)  # Best-practice? Expensive...
omega.SetIncludeInput(False)
omega.SetStrictStereo(True)  # Refuse to generate conformers if stereochemistry not provided

from openforcefield.typing.engines.smirnoff import ForceField, generateTopologyFromOEMol
from openeye import oechem  # OpenEye Python toolkits

from openeye import oequacpac  # Charge toolkit

ff = ForceField('forcefield/smirnoff99Frosst.ffxml')


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


# make a list of just the SMILES strings in the database
smiles = [entry[1] for entry in db]

# parallelize over compounds
pool = Pool(n_subprocesses)
generated_systems = pool.map(generate_mol_top_sys_pos, smiles)

from pickle import dump

with open('generated_systems.pkl', 'wb') as f:
    dump(generated_systems, f)

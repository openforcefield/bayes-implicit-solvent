# prepares SMIRNOFF'd openmm systems ready to simulate, saves the serialized systems in a pandas dataframe

from pkg_resources import resource_filename

path_to_freesolv = resource_filename('bayes_implicit_solvent', 'data/FreeSolv-0.52/database.txt')
path_to_mol_top_sys_pos_list = resource_filename('bayes_implicit_solvent', 'data/mol_top_sys_pos.pkl')
path_to_smiles_list = resource_filename('bayes_implicit_solvent', 'data/smiles.pkl')

from glob import glob

with open(path_to_freesolv, 'r') as f:
    freesolv = f.read()
db = []
for entry in freesolv.split('\n')[3:-1]:
    db.append(entry.split('; '))
smiles_list = [entry[1] for entry in db]
cid_list = [entry[0] for entry in db]
smiles_to_cid = dict(zip(smiles_list, cid_list))
cid_to_smiles = dict(zip(cid_list, smiles_list))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.mol2')
    return path[i:j]


def prepare_freesolv():

    path_to_mol2_files = resource_filename('bayes_implicit_solvent', 'data/FreeSolv-0.52/mol2files_gaff/*')
    mol2_paths = glob(path_to_mol2_files)
    print('example path to mol2 file: ', mol2_paths[0])

    from tqdm import tqdm

    print('example freesolv id extracted from mol2 filepath: ', extract_cid_key(mol2_paths[0]))

    from openeye import oechem


    def load_oe_graph_mol(mol2_filepath):
        """Load a single OpenEye molecule from a mol2 file.

        References
        ----------
        * I copied this verbatim from: https://github.com/MobleyLab/SMIRNOFF_paper_code/blob/3e09278207b408b5d7030c971ff555e2346d77d8/FreeSolv/scripts/create_input_files.py#L51
        """
        # OpenEye flavors to use when loading mol2 files.
        flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield

        ifs = oechem.oemolistream(mol2_filepath)
        ifs.SetFlavor(oechem.OEFormat_MOL2, flavor)
        mol = oechem.OEGraphMol()

        molecules = []
        while oechem.OEReadMolecule(ifs, mol):
            oechem.OETriposAtomNames(mol)
            molecules.append(oechem.OEGraphMol(mol))

        # The script assumes we have only 1 molecule per mol2 file.
        assert len(molecules) == 1
        return molecules[0]

    print('loading oemols')
    oemols = {}
    for path in mol2_paths:
        key = extract_cid_key(path)
        oemols[key] = load_oe_graph_mol(path)

    legend = freesolv.split('\n')[2].split('; ')
    print('FreeSolv column labels')
    print(list(zip(range(len(legend)), legend)))

    import numpy as np

    from openforcefield.typing.engines.smirnoff import ForceField, generateTopologyFromOEMol
    import openforcefield

    print(openforcefield._version.get_versions())

    ff = ForceField(resource_filename('bayes_implicit_solvent', 'data/smirnoff99Frosst.offxml'))


    def fetch_oemol(smiles):
        cid_key = smiles_to_cid[smiles]
        return oemols[cid_key]


    def generate_mol_top_sys_pos(smiles):
        """Generate an openmm topology, openmm system, and coordinate array from a smiles string"""
        mol = fetch_oemol(smiles)

        coord_dict = mol.GetCoords()
        positions = np.array([coord_dict[key] for key in coord_dict])

        topology = generateTopologyFromOEMol(mol)
        system = ff.createSystem(topology, [mol])

        return mol, topology, system, positions


    mol_top_sys_pos_list = []

    for smiles in tqdm(smiles_list):
        mol_top_sys_pos_list.append(tuple(generate_mol_top_sys_pos(smiles)))

    from pickle import dump

    # save list of (mol, topology, system, positions) tuples, in the order of the sorted smiles list
    with open(path_to_mol_top_sys_pos_list, 'wb') as f:
        dump(mol_top_sys_pos_list, f)

    # save the smiles list
    with open(path_to_smiles_list, 'wb') as f:
        dump(smiles_list, f)

import os.path
from pickle import load
if (not os.path.isfile(path_to_mol_top_sys_pos_list)) or (not os.path.isfile(path_to_smiles_list)):
    prepare_freesolv()

with open(path_to_mol_top_sys_pos_list, 'rb') as f:
    mol_top_sys_pos_list = load(f)

with open(path_to_smiles_list, 'rb') as f:
    smiles_list = load(f)

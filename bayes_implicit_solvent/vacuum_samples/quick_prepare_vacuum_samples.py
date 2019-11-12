from bayes_implicit_solvent.freesolv import smiles_list, mol_top_sys_pos_list, smiles_to_cid
from bayes_implicit_solvent.solvation_free_energy import get_vacuum_samples
import numpy as np
from simtk import unit

n_samples = 50
thinning = 1000
from tqdm import tqdm

# removing mdtraj because now I'm getting tons of errors like
# File "...anaconda3/lib/python3.7/site-packages/mdtraj/core/topology.py", line 403, in from_openmm
#     out.add_bond(atom_mapping[a1], atom_mapping[a2], type=bond_mapping[bond.type], order=bond.order)
# KeyError: 'Aromatic'
# whenever I try to use `md.Topology().from_openmm(top)`

if __name__ == '__main__':
    for i in tqdm(range(len(mol_top_sys_pos_list))):
        mol, top, sys, pos = mol_top_sys_pos_list[i]
        smiles = smiles_list[i]

        vacuum_sim, vacuum_traj = get_vacuum_samples(top, sys, pos,
                                                     n_samples=n_samples, thinning=thinning)

        xyz = np.array([snapshot / unit.nanometer for snapshot in vacuum_traj])

        fname = 'vacuum_samples_{}.npy'.format(smiles_to_cid[smiles])
        np.save(fname, xyz)

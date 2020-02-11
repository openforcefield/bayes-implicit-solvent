from bayes_implicit_solvent.solvation_free_energy import get_vacuum_samples
from bayes_implicit_solvent.freesolv import mol_top_sys_pos_list
import numpy as np
import mdtraj as md
from simtk import unit

n_samples = 100
thinning = 1000


if __name__ == '__main__':

    from time import time
    t0 = time()
    for i in range(len(mol_top_sys_pos_list)):

        print('molecule {} of {}'.format(i, len(mol_top_sys_pos_list)))
        print('total elapsed time: {:.4f}s'.format(time() - t0))

        mol, top, sys, pos = mol_top_sys_pos_list[i]

        vacuum_sim, vacuum_traj = get_vacuum_samples(top, sys, pos,
                                                     n_samples=n_samples, thinning=thinning)

        xyz = np.array([snapshot / unit.nanometer for snapshot in vacuum_traj])
        vacuum_traj = md.Trajectory(xyz, md.Topology().from_openmm(top))
        vacuum_traj.save_hdf5('vacuum_samples_{}.h5'.format(i))

from bayes_implicit_solvent.solvation_free_energy import get_vacuum_samples, mol_top_sys_pos_list
import numpy as np
import mdtraj as md
from simtk import unit

n_samples = 10000
thinning = 10000
burn_in_steps = 10 * thinning

all_problematic_indices = [11, 19, 27, 35, 37, 44, 48, 71, 80, 89, 116, 139, 154, 157, 178, 217, 221, 242, 258, 259, 303, 344, 350, 387, 411, 415, 451, 452, 459, 502, 504, 522]


if __name__ == '__main__':
    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No valid job_id supplied! Selecting one at random")
        job_id = np.random.randint(len(all_problematic_indices)) + 1

    system_id = all_problematic_indices[job_id - 1]
    print('job_id', job_id)
    print('system_id', system_id)

    mol, top, sys, pos = mol_top_sys_pos_list[system_id]

    vacuum_sim, vacuum_traj = get_vacuum_samples(top, sys, pos,
                                                 n_samples=n_samples, thinning=thinning)

    xyz = np.array([snapshot / unit.nanometer for snapshot in vacuum_traj])
    vacuum_traj = md.Trajectory(xyz, md.Topology().from_openmm(top))
    vacuum_traj.save_hdf5('extended_vacuum_samples_{}.h5'.format(system_id))

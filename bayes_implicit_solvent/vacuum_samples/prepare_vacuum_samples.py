from bayes_implicit_solvent.solvation_free_energy import get_vacuum_samples, mol_top_sys_pos_list
import numpy as np


n_samples = 10000
thinning = 1000


if __name__ == '__main__':
    import sys

    try:
        job_id = int(sys.argv[1])
    except:
        print("No valid job_id supplied! Selecting one at random")
        job_id = np.random.randint(len(mol_top_sys_pos_list)) + 1

    print(job_id)
    mol, top, sys, pos = mol_top_sys_pos_list[job_id - 1]

    vacuum_sim, vacuum_traj = get_vacuum_samples(top, sys, pos,
                                                 n_samples=n_samples, thinning=thinning)

    vacuum_traj.save_hdf5('vacuum_samples_{}.h5')

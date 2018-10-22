"""In this file we'll

"""

import numpy as np
from bayes_implicit_solvent.posterior_sampling import Molecule
from bayes_implicit_solvent.samplers import random_walk_mh
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

from pkg_resources import resource_filename
import mdtraj as md


def methane_demo(n_configuration_samples=10, n_parameter_samples=100000):
    """Run toy 2D parameterization demo with methane only"""

    np.random.seed(0)

    smiles = 'C'
    mol = Molecule(smiles, vacuum_samples=[])
    path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                               'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                   mol.mol_index_in_smiles_list))
    vacuum_traj = md.load(path_to_vacuum_samples)
    thinning = int(len(vacuum_traj) / n_configuration_samples)
    mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
    print('thinned vacuum_traj from {} to {}'.format(len(mol.vacuum_traj), len(mol.vacuum_traj)))

    def log_prob(radii):
        atomic_radii = np.zeros(len(mol.pos))
        atomic_radii[0] = radii[0]
        atomic_radii[1:] = radii[1]
        return mol.log_prob(atomic_radii)

    radii0 = np.array([0.1,0.1])

    traj, log_probs, acceptance_fraction = random_walk_mh(radii0, log_prob,
                                                          n_steps=n_parameter_samples, stepsize=0.1)
    import os.path

    data_path = '../data/'
    np.save(os.path.join(data_path, 'H_vs_not_radii_samples_{}.npy'.format(smiles)), traj)

    print("acceptance fraction: {:.4f}".format(acceptance_fraction))

if __name__ == '__main__':
    methane_demo()
"""In this file we'll show that at one extreme, if we have a very poor / inflexible typing scheme,
we'll have a hard time finding any setting of the continuous parameters that reproduces the desired
measurements.

Our toy model will contain just two types -- "hydrogen" vs "not hydrogen" -- so we can plot in 2D
and inspect what's happening.

We'll attempt to fit GB radii using this restricted typing scheme on the following subsets of FreeSolv:
* A single molecule (methane)
* A sequence of alkanes up to length 10
* A randomly selected quarter of FreeSolv
* All of FreeSolv

We can also see how the results depend on the number of configuration-samples used in our hydration free energy estimates.
"""

import os.path

import mdtraj as md
import numpy as np
from pkg_resources import resource_filename

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.samplers import random_walk_mh
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'


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
    print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))

    def log_prob(radii):
        atomic_radii = np.zeros(len(mol.pos))
        atomic_radii[0] = radii[0]
        atomic_radii[1:] = radii[1]

        # TODO: update this example to allow the scaling_factors to be variable also
        default_scaling_factors = np.ones(len(radii))

        return mol.log_prob(atomic_radii, default_scaling_factors)

    radii0 = np.array([0.1, 0.1])

    traj, log_probs, acceptance_fraction = random_walk_mh(radii0, log_prob,
                                                          n_steps=n_parameter_samples, stepsize=0.1)

    np.save(os.path.join(data_path, 'H_vs_not_radii_samples_{}.npy'.format(smiles)), traj)

    print("acceptance fraction: {:.4f}".format(acceptance_fraction))


def alkanes_demo(n_configuration_samples=100, n_parameter_samples=10000):
    """Run toy 2D parameterization demo with alkanes of length 1 through 10"""

    np.random.seed(0)

    alkanes = ['C' * i for i in range(1, 11)]

    mols = []
    hydrogens = []

    for smiles in alkanes:
        mol = Molecule(smiles, vacuum_samples=[])
        path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                                   'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                       mol.mol_index_in_smiles_list))
        vacuum_traj = md.load(path_to_vacuum_samples)
        thinning = int(len(vacuum_traj) / n_configuration_samples)
        mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
        print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))
        hydrogens.append(np.array([a.element.symbol == 'H' for a in mol.top.atoms()]))
        mols.append(mol)

    def log_prob(radii):
        logp = 0
        for i in range(len(mols)):
            mol = mols[i]
            atomic_radii = np.ones(len(mol.pos)) * radii[0]
            atomic_radii[hydrogens[i]] = radii[1]

            # TODO: update this example to allow the scaling_factors to be variable also
            default_scaling_factors = np.ones(len(mol.pos))
            logp += mol.log_prob(atomic_radii, default_scaling_factors)

        return logp

    radii0 = np.array([0.1, 0.1])

    traj, log_probs, acceptance_fraction = random_walk_mh(radii0, log_prob,
                                                          n_steps=n_parameter_samples, stepsize=0.005)

    np.save(os.path.join(data_path, 'H_vs_not_radii_samples_alkanes_n_config={}.npy'.format(n_configuration_samples)),
            traj)

    print("acceptance fraction: {:.4f}".format(acceptance_fraction))


def pack(radii, scales):
    n = len(radii)
    theta = np.zeros(2 * n)
    theta[:n] = radii
    theta[n:2 * n] = scales
    return theta


def unpack(theta):
    n = int((len(theta)) / 2)
    radii, scales = theta[:n], theta[n:2 * n]
    return radii, scales

def quarter_freesolv_demo(n_configuration_samples=10, n_parameter_samples=10000, good_initialization=False):
    """Run toy 2D parameterization demo with one randomly-selected quarter of freesolv"""

    np.random.seed(0)

    inds = np.arange(len(smiles_list))
    np.random.shuffle(inds)
    inds = inds[:int(len(smiles_list) / 4)]

    quarter_smiles = [smiles_list[i] for i in inds]

    mols = []
    hydrogens = []

    for smiles in quarter_smiles:
        mol = Molecule(smiles, vacuum_samples=[])
        path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                                   'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                       mol.mol_index_in_smiles_list))
        vacuum_traj = md.load(path_to_vacuum_samples)
        thinning = int(len(vacuum_traj) / n_configuration_samples)
        mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
        print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))
        hydrogens.append(np.array([a.element.symbol == 'H' for a in mol.top.atoms()]))
        mols.append(mol)

    def log_prob(theta):
        radii, scales = unpack(theta)
        logp = 0
        for i in range(len(mols)):
            mol = mols[i]
            atomic_radii = np.ones(len(mol.pos)) * radii[0]
            atomic_radii[hydrogens[i]] = radii[1]

            atomic_scales = np.ones(len(mol.pos)) * scales[0]
            atomic_scales[hydrogens[i]] = scales[1]

            logp += mol.log_prob(atomic_radii, atomic_scales)

        return logp


    radii0 = np.array([0.1, 0.1])
    scales0 = np.array([0.8, 0.8])
    if good_initialization:
        radii0 = np.array([0.28319081, 0.20943347])
        scales0 = np.array([0.89298609, 0.67449963])

    theta0 = pack(radii0, scales0)

    stepsize = 0.0005

    traj, log_probs, acceptance_fraction = random_walk_mh(theta0, log_prob,
                                                          n_steps=n_parameter_samples, stepsize=stepsize)

    np.savez(os.path.join(data_path,
                          'H_vs_not_freesolv_{}_dt={}.npz'.format(len(quarter_smiles), stepsize)),
             traj=traj, log_probs=log_probs, acceptance_fraction=acceptance_fraction, stepsize=stepsize,
             n_steps=n_parameter_samples, smiles_subset=quarter_smiles, n_configuration_samples=n_configuration_samples)

    print("acceptance fraction: {:.4f}".format(acceptance_fraction))


def freesolv_demo(n_configuration_samples=10, n_parameter_samples=10000):
    """Run toy 2D parameterization demo with all of freesolv"""

    np.random.seed(0)

    mols = []
    hydrogens = []

    for smiles in smiles_list:
        mol = Molecule(smiles, vacuum_samples=[])
        path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                                   'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                       mol.mol_index_in_smiles_list))
        vacuum_traj = md.load(path_to_vacuum_samples)
        thinning = int(len(vacuum_traj) / n_configuration_samples)
        mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
        print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))
        hydrogens.append(np.array([a.element.symbol == 'H' for a in mol.top.atoms()]))
        mols.append(mol)

    def log_prob(radii):
        logp = 0
        for i in range(len(mols)):
            mol = mols[i]
            atomic_radii = np.ones(len(mol.pos)) * radii[0]
            atomic_radii[hydrogens[i]] = radii[1]

            # TODO: update this example to allow the scaling_factors to be variable also
            default_scaling_factors = np.ones(len(mol.pos))
            logp += mol.log_prob(atomic_radii, default_scaling_factors)

        return logp

    radii0 = np.array([0.1, 0.1])
    scales0 = np.array([0.8, 0.8])
    theta0 = pack(radii0, scales0)

    stepsize = 0.002

    traj, log_probs, acceptance_fraction = random_walk_mh(theta0, log_prob,
                                                          n_steps=n_parameter_samples, stepsize=stepsize)


    np.savez(os.path.join(data_path,
                          'H_vs_not_freesolv.npz'),
             traj=traj, log_probs=log_probs, acceptance_fraction=acceptance_fraction, stepsize=stepsize,
             n_steps=n_parameter_samples)

    print("acceptance fraction: {:.4f}".format(acceptance_fraction))


if __name__ == '__main__':
    #methane_demo()

    #for n_config in [10, 50, 100]:
    #    alkanes_demo(n_configuration_samples=n_config)

    quarter_freesolv_demo(n_configuration_samples=5, n_parameter_samples=1000, good_initialization=True)
    #freesolv_demo(n_configuration_samples=10, n_parameter_samples=1000)

import os.path

import numpy as np
from bayes_implicit_solvent.molecule import Molecule
from simtk import unit


def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = np.load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj


from glob import glob
from pkg_resources import resource_filename

data_path = resource_filename('bayes_implicit_solvent',
                              'data')
ll = 'gaussian'  # or 'student-t'
randomize_theta0 = False
n_conf = 50

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)
print('number of molecules being considered: {}'.format(len(paths_to_samples)))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))

mols = []

n_configuration_samples = n_conf  # TODO: Since this is cheaper, can probably modify this a bit...

name = 'n_config={}_{}_ll'.format(n_configuration_samples, ll)

from bayes_implicit_solvent.freesolv import cid_to_smiles

for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning], ll=ll)
    mols.append(mol)

# 2. Define a likelihood function, including "type-assignment"

from bayes_implicit_solvent.gb_models.obc2_parameters import obc2_model

type_slices = [obc2_model.apply_to_molecule(mol.mol) for mol in mols]


def construct_arrays(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]

    parameterized_list = []
    for i in range(len(mols)):
        parameterized_list.append((radii[type_slices[i]], scales[type_slices[i]]))
    return parameterized_list


from bayes_implicit_solvent.constants import min_r, max_r, min_scale, max_scale


def log_prior(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]

    if (min(radii) < min_r) or (max(radii) > max_r) or (min(scales) < min_scale) or (max(scales) > max_scale):
        return - np.inf
    else:
        return 0


def log_prob(theta):
    L = log_prior(theta)
    if not (L > -np.inf):
        return L
    else:
        parameterized_list = construct_arrays(theta)
        for i, mol in enumerate(mols):
            radii, scale_factors = parameterized_list[i]
            L += mol.log_prob(radii, scale_factors)
    return L


if __name__ == '__main__':
    n_types = obc2_model.number_of_nodes
    print('n_types: {}'.format(n_types))

    initial_radii = np.array(obc2_model.get_radii())
    initial_scales = np.array(obc2_model.get_scale_factors())
    theta0 = np.hstack((initial_radii, initial_scales))

    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)

    minimized_theta_fname = os.path.join(data_path,
                                         'elemental_types_l-bfgs-finite-difference_{}.npy'.format(
                                             name))

    print('minimizing...')
    from scipy.optimize import minimize

    bounds = [(min_r, max_r)] * n_types + [(min_scale, max_scale)] * n_types

    traj = []
    loss_traj = []


    def loss(theta):
        L = - log_prob(theta)
        print('loss at {} = {}'.format(theta, L))
        traj.append(theta)
        loss_traj.append(L)
        return L


    print('starting minimization')
    result = minimize(loss, theta0,
                      method='Nelder-Mead',
                      bounds=bounds,
                      options={'disp': True,
                               'maxiter': 1000,
                               })

    traj_fname = os.path.join(data_path, 'elemental_types_nelder-mead_freesolv_{}_traj.npz'.format(
        name))
    np.savez(traj_fname, traj=traj, loss_traj=loss_traj)

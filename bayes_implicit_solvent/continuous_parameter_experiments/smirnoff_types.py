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
n_conf = 25

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

from bayes_implicit_solvent.gb_models.smirnoff99frosst_nb_types import typer as initial_model

type_slices = [initial_model.apply_to_molecule(mol.mol) for mol in mols]


def construct_arrays(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]

    parameterized_list = []
    for i in range(len(mols)):
        parameterized_list.append((radii[type_slices[i]], scales[type_slices[i]]))
    return parameterized_list


from bayes_implicit_solvent.constants import min_r, max_r, min_scale, max_scale
from scipy.stats import norm

expected_radius = 0.15
expected_scale = 0.8

radius_sigma = 1.0
scale_sigma = 1.0


def log_prior_on_radii(radii):
    return sum(norm.logpdf(radii, loc=expected_radius, scale=radius_sigma))


def log_prior_on_scales(scales):
    return sum(norm.logpdf(scales, loc=expected_scale, scale=scale_sigma))


def log_prior(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]

    if (min(radii) < min_r) or (max(radii) > max_r) or (min(scales) < min_scale) or (max(scales) > max_scale):
        return - np.inf
    else:
        return log_prior_on_radii(radii) + log_prior_on_scales(scales)


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
    n_types = initial_model.number_of_nodes
    print('n_types: {}'.format(n_types))

    # initial_radii = np.array(initial_model.get_radii())
    # initial_scales = np.array(initial_model.get_scale_factors())
    # theta0 = np.hstack((initial_radii, initial_scales))

    theta0 = np.array(list(map(float, '0.18544694 0.13589401 0.12459711 0.15561623 0.13462232 0.1597141 0.16367472 0.16122833 0.18471579 0.12737076 0.15901924 0.13017523 0.14424222 0.15360279 0.14665745 0.14714415 0.13652498 0.13477897 0.15213938 0.14851908 0.16365621 0.12496874 0.11879516 0.13049311 0.14916848 0.14840174 0.86159337 0.75604993 0.64205759 0.90736801 0.77463168 0.82812836 0.86548253 0.9858935  0.83913711 0.95611967 0.80626157 0.79300531 0.9533121  0.8106477  0.67812537 0.76018205 0.80109448 0.83501951 0.787929   0.68157676 0.91278139 0.78743348 0.94062257 0.85178941 0.77598672 0.79416221'.split())))

    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)

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

    traj_fname = os.path.join(data_path, 'smirnoff_nb_types_initialized_nelder-mead_freesolv_{}_traj.npz'.format(
        name))
    np.savez(traj_fname, traj=traj, loss_traj=loss_traj)

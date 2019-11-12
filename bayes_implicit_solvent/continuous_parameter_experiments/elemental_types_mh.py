import os.path

import numpy as np
from bayes_implicit_solvent.molecule import Molecule
from simtk import unit

from bayes_implicit_solvent.samplers import sparse_mh


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
n_conf = 25

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/short_run/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)
np.random.seed(0)
np.random.shuffle(paths_to_samples)
paths_to_samples = paths_to_samples[::2]

print('number of molecules being considered: {}'.format(len(paths_to_samples)))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))
print('first few CIDs', cids[:5])

mols = []

n_configuration_samples = n_conf  # TODO: Since this is cheaper, can probably modify this a bit...

name = 'n_config={}_{}_ll'.format(n_configuration_samples, ll)

from bayes_implicit_solvent.freesolv import cid_to_smiles

from bayes_implicit_solvent.constants import beta
def unreduce(value):
    """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
    return value / (beta * unit.kilocalorie_per_mole)

for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning], ll=ll)
    if (unreduce(mol.experimental_value) > -15) and (unreduce(mol.experimental_value) < 5):
        mols.append(mol)
    else:
        print('discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles, cid))

# 2. Define a likelihood function, including "type-assignment"

from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

type_slices = [mbondi_model.apply_to_molecule(mol.mol) for mol in mols]


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
    n_types = mbondi_model.number_of_nodes
    print('n_types: {}'.format(n_types))

    initial_radii = np.array(mbondi_model.get_radii())
    initial_scales = np.array(mbondi_model.get_scale_factors())
    theta0 = np.hstack((initial_radii, initial_scales))

    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)

    minimized_theta_fname = os.path.join(data_path,
                                         'elemental_types_l-bfgs-finite-difference_{}.npy'.format(
                                             name))
    stepsize = 0.5 * 1e-1
    n_steps = 10000
    dim_to_perturb = 5

    traj, log_probs, acceptance_fraction = sparse_mh(theta0, log_prob, n_steps=n_steps, stepsize=stepsize,
                                                     dim_to_perturb=dim_to_perturb)

    np.savez(os.path.join(data_path,
                          'elemental_types_mh_freesolv_{}.npz'.format(
                              name)),
             traj=traj, log_probs=log_probs, acceptance_fraction=acceptance_fraction, stepsize=stepsize,
             n_steps=n_steps, dim_to_perturb=dim_to_perturb, cids=cids, n_configuration_samples=n_conf)

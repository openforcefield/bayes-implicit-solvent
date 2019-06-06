# use unadjusted langevin to sample the radii and scales in
# the elemental-types-only model


import os.path

import mdtraj as md
from autograd import numpy as np
from pkg_resources import resource_filename
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.samplers import langevin
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'
gaussian_ll = True

import sys

try:
    job_id = int(sys.argv[1])
    print("Using supplied job_id: ", job_id)
except:
    job_id = np.random.randint(1000)
    print("No valid job_id supplied! Selecting one at random: ", job_id)

# 1. load the dataset, precompute distance matrices
np.random.seed(job_id)

inds = np.arange(len(smiles_list))[::5]
#np.random.shuffle(inds)
#inds = inds[:int(len(smiles_list) * 0.5)]
smiles_subset = [smiles_list[i] for i in inds]

mols = []
elements = []
charges = []
distance_matrices = []
expt_means = []
expt_uncs = []
vacuum_trajs = []

n_configuration_samples = 5  # TODO: Since this is cheaper, can probably modify this a bit...

name = 'n_config={}_job_id={}'.format(n_configuration_samples, job_id)
if gaussian_ll:
    name = name + '_gaussian_ll'
smiles_subset_fname = os.path.join(data_path,
                                   'smiles_subset_{}.txt'.format(name))
with open(smiles_subset_fname, 'w') as f:
    f.writelines(['{}\n'.format(s) for s in smiles_subset])

from bayes_implicit_solvent.utils import get_charges
from scipy.spatial.distance import pdist, squareform

for smiles in smiles_subset:
    mol = Molecule(smiles, vacuum_samples=[])
    path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                               'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                   mol.mol_index_in_smiles_list))
    vacuum_traj = md.load(path_to_vacuum_samples)
    thinning = int(len(vacuum_traj) / n_configuration_samples)
    mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
    vacuum_trajs.append(mol.vacuum_traj)
    print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))

    expt_means.append(mol.experimental_value)
    expt_uncs.append(mol.experimental_uncertainty)

    elements.append(np.array([a.element.atomic_number for a in mol.top.atoms()]))
    charges.append(get_charges(mol.sys))
    distance_matrices.append([squareform(pdist(snapshot / unit.nanometer)) for snapshot in mol.vacuum_traj])
    mols.append(mol)
# 2. Define a likelihood function, including "type-assignment"
from autograd import numpy as np
from autograd.scipy.stats import norm
from autograd.scipy.stats import t as student_t
from autograd import grad
from autograd.scipy.misc import logsumexp
from simtk import unit
from bayes_implicit_solvent.constants import kB, temperature

kj_mol_to_kT = 1.0 * unit.kilojoule_per_mole / (kB * temperature)


def one_sided_exp(w_F):
    DeltaF = - (logsumexp(- w_F) - np.log(len(w_F)))
    return DeltaF


# TODO: parallelize using multiprocessing

all_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


#element_dict = {}
#for a in all_elements:
#    element_dict[a] = 0
#element_dict[1] = 1

element_dict = dict(zip(all_elements, list(range(len(all_elements)))))


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


def construct_array(i, theta):
    radii, scales = unpack(theta)

    mol_radii = np.array([radii[element_dict[element]] for element in elements[i]])
    mol_scales = np.array([scales[element_dict[element]] for element in elements[i]])
    return mol_radii, mol_scales


def construct_arrays(theta):
    arrays = [construct_array(i, theta) for i in range(len(mols))]
    mol_radii = [a[0] for a in arrays]
    mol_scales = [a[1] for a in arrays]
    return mol_radii, mol_scales


from bayes_implicit_solvent.gb_models.numpy_gb_models import compute_OBC_energy_vectorized

def log_prob(theta):
    mol_radii, mol_scales = construct_arrays(theta)
    if min(theta) < 0.001 or max(theta) > 2:
        print('out of bounds!')
        return -np.inf
    logp = 0
    for i in range(len(mols)):
        radii = mol_radii[i]
        scales = mol_scales[i]
        W_F = np.array([compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges[i]) for distance_matrix in
                        distance_matrices[i]])
        w_F = W_F * kj_mol_to_kT
        pred_free_energy = one_sided_exp(w_F)

        if gaussian_ll:
            logp += norm.logpdf(pred_free_energy, loc=expt_means[i], scale=expt_uncs[i] ** 2)
        else:
            logp += student_t.logpdf(pred_free_energy, loc=expt_means[i],
                                     scale=expt_uncs[i],
                                     df=7)
    return logp

grad_log_prob = grad(log_prob)

# 3. Take gradient of likelihood function

if __name__ == '__main__':
    n_types = len(set(element_dict.values()))


    # 4. Minimize for a few steps

    initial_radii = np.ones(n_types) * 0.12
    initial_scales = np.ones(n_types) * 0.85

    theta0 = pack(initial_radii, initial_scales)
    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)
    print('initial log prob', initial_log_prob)
    print('initial gradient of log prob', grad_log_prob(theta0))

    print('initial gradient norm = {}'.format(np.linalg.norm(grad_log_prob(theta0))))

    minimized_theta_fname = os.path.join(data_path,
                                         'elemental_types_l-bfgs_freesolv_{}.npy'.format(
                                             name))

    print('minimizing...')
    from scipy.optimize import minimize

    loss = lambda theta: - log_prob(theta)
    grad_loss = lambda theta: - grad_log_prob(theta)
    bounds = [(0.001, 2.0)] * len(theta0)
    result = minimize(loss, theta0,
                  jac=grad(loss),
                  method='L-BFGS-B',
                  bounds=bounds,
                  options={'disp': True,
                           'maxiter': 5})

    theta1 = result.x
    np.save(minimized_theta_fname, theta1)
    print('theta after initial minimization', theta1)
    print('gradient norm after initial minimization = {}'.format(np.linalg.norm(grad_log_prob(theta1))))

    # 5. Run MALA

    stepsize = 5e-7
    collision_rate = 1e-1
    n_steps = 1000
    traj, log_probs = langevin(x0=theta1,
                               v0=np.random.randn(len(theta1)),
                               log_prob_fun=log_prob, grad_log_prob_fun=grad_log_prob,
                               n_steps=n_steps, stepsize=stepsize, collision_rate=collision_rate)

    np.savez(os.path.join(data_path,
                          'elemental_types_langevin_freesolv_{}.npz'.format(
                              name)),
             traj=traj, stepsize=stepsize, log_probs=log_probs, collision_rate=collision_rate, n_steps=n_steps, gaussian_ll=gaussian_ll)

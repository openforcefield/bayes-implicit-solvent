# use metropolis-adjusted langevin to sample the radii and scales in
# the elemental-types-only model


import os.path

import mdtraj as md
import numpy as np
from pkg_resources import resource_filename
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.samplers import MALA
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'
gaussian_ll = True

import sys

try:
    job_id = int(sys.argv[1])
    print("Using supplied job_id: ", job_id)
except:
    #job_id = random.randint(key=key, shape=(1,), minval=0, maxval=1000)
    job_id = np.random.randint(1000)
    print("No valid job_id supplied! Selecting one at random: ", job_id)

# 1. load the dataset, precompute distance matrices
np.random.seed(job_id)

inds = np.arange(len(smiles_list))
np.random.shuffle(inds)
#random.shuffle(key, inds)
inds = inds[:5] #inds[:int(len(smiles_list) * 0.05)]
smiles_subset = [smiles_list[i] for i in inds]

mols = []
elements = []
charges = []
distance_matrices = []
expt_means = []
expt_uncs = []
vacuum_trajs = []

n_configuration_samples = 2  # TODO: Since this is cheaper, can probably modify this a bit...

name = 'n_config={}_job_id={}'.format(n_configuration_samples, job_id)
if gaussian_ll:
    name = name + '_gaussian_ll'
    ll = 'gaussian'
else:
    ll = 'student-t'
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
    distance_matrices.append(np.array([squareform(pdist(snapshot / unit.nanometer)) for snapshot in mol.vacuum_traj]))
    mols.append(mol)
#distance_matrices = np.array(distance_matrices)
# 2. Define a likelihood function, including "type-assignment"
from jax import numpy as np
from jax.scipy.stats import norm
#from jax.scipy.stats import t as student_t
from jax import grad
from jax.scipy.misc import logsumexp
from simtk import unit
from bayes_implicit_solvent.constants import kB, temperature

kj_mol_to_kT = 1.0 * unit.kilojoule_per_mole / (kB * temperature)


def one_sided_exp(w_F):
    DeltaF = - (logsumexp(- w_F) - np.log(len(w_F)))
    return DeltaF


# TODO: parallelize using multiprocessing

all_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
element_dict = dict(zip(all_elements, list(range(len(all_elements)))))


def pack(radii, scales):
    return np.hstack((radii, scales))


def unpack(theta):
    n = int((len(theta)) / 2)
    radii, scales = theta[:n], theta[n:2 * n]
    return radii, scales


slices = [np.array([int(element_dict[int(element)]) for element in elements[i]]) for i in range(len(elements))]
print(slices[0])
def construct_array(i, theta):
    radii, scales = unpack(theta)
    K0 = len(radii)
    K1 = len(scales)
    K2 = len(slices[i])
    if (K0 != K1):
        print('lengths dont agree!')
        print('len(radii), len(scales), len(slices[i])', K0, K1, K2)
    print('radii[slices[i]]', radii[slices[i]])
    return radii[slices[i]], scales[slices[i]]

from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
#from jax import jit

#@jit
def log_prob_component(i, theta):
    radii, scales = construct_array(i, theta)

    #@jit
    def component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges[i])

    # TODO: vmap
    W_F = np.array(list(map(component, distance_matrices[i])))
    #W_F = vmap(component)(distance_matrices[i])
    w_F = W_F * kj_mol_to_kT
    pred_free_energy = one_sided_exp(w_F)
    if gaussian_ll:
        return norm.logpdf(pred_free_energy, loc=expt_means[i], scale=expt_uncs[i] ** 2)
    else:
        # TODO : fix
        # https://github.com/scipy/scipy/blob/c3fa90dcfcaef71658744c73578e9e7d915c81e9/scipy/stats/_continuous_distns.py#L5207
        # def _logpdf(self, x, df):
        #         r = df*1.0
        #         lPx = sc.gammaln((r+1)/2)-sc.gammaln(r/2)
        #         lPx -= 0.5*np.log(r*np.pi) + (r+1)/2*np.log(1+(x**2)/r)
        #         return lPx
        raise(NotImplementedError)

        return student_t.logpdf(pred_free_energy, loc=expt_means[i],
                                scale=expt_uncs[i] ** 2,
                                df=7)

def log_prob(theta):

    if min(theta[:-1]) < 0.001 or max(theta[:-1]) > 2:
        print('out of bounds!')
        return -np.inf

    def logp_(i):
        return log_prob_component(i, theta)

    logps = list(map(logp_, np.arange(len(mols))))
    return sum(logps)

# 3. Take gradient of likelihood function
from jax import numpy as np
from jax import grad, vmap
import jax.random as random
key = random.PRNGKey(0)

if __name__ == '__main__':

    # 4. Minimize for a few steps
    initial_radii = np.ones(len(all_elements)) * 0.12
    initial_scales = np.ones(len(all_elements)) * 0.85

    theta0 = pack(initial_radii, initial_scales)
    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)
    print('initial log prob', initial_log_prob)

    # TODO: potentially jit() this
    grad_log_prob = grad(log_prob)
    #grad_log_prob = parallel_grad_log_prob

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
                  jac=grad_loss,
                  method='L-BFGS-B',
                  bounds=bounds,
                  options={'disp': True,
                           'maxiter': 5})

    theta1 = result.x
    np.save(minimized_theta_fname, theta1)
    print('theta after initial minimization', theta1)
    print('gradient norm after initial minimization = {}'.format(np.linalg.norm(grad_log_prob(theta1))))

    # 5. Run MALA

    stepsize = 1e-8
    n_steps = 2000
    traj, log_probs, grads, acceptance_probabilities, stepsizes = MALA(theta1, log_prob, grad_log_prob,
                                                                       n_steps=n_steps, stepsize=stepsize,
                                                                       adapt_stepsize=True)

    np.savez(os.path.join(data_path,
                          'elemental_types_mala_freesolv_{}.npz'.format(
                              name)),
             traj=traj, grads=grads, acceptance_probabilities=acceptance_probabilities, stepsizes=stepsizes)

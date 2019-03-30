
from bayes_implicit_solvent.molecule import Molecule
from numpy import load
from simtk import unit

def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj

ll = 'gaussian'
from scipy.stats import t as student_t

from glob import glob
from pkg_resources import resource_filename

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)
from numpy import random
random.seed(0)
random.shuffle(paths_to_samples)
paths_to_samples = paths_to_samples[::2]

print('number of molecules being considered: {}'.format(len(paths_to_samples)))


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))
print('first few CIDs', cids[:5])

mols = []

n_configuration_samples = 15

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
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning])

    if (unreduce(mol.experimental_value) > -15) and (unreduce(mol.experimental_value) < 5):
        mols.append(mol)
    else:
        print('discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles, cid))

oemols = [mol.mol for mol in mols]
import numpy as np

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp

@jit
def predict_solvation_free_energy_jax(radii, scaling_factors, distance_matrices, charges):

    @jit
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)

    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)

distance_matrices = [mol.distance_matrices for mol in mols]
charges = [mol.charges for mol in mols]

expt_means = unreduce(np.array([mol.experimental_value for mol in mols]))
expt_uncs = unreduce(np.array([mol.experimental_uncertainty for mol in mols]))

from jax.scipy.stats import norm

radius_prior = 0.15
scale_prior = 0.8

N_types = 5
from numpy import random
dummy_types = [random.randint(0, N_types, mol.mol.NumAtoms()) for mol in mols]

dummy_theta = np.hstack([np.ones(N_types) * radius_prior, np.ones(N_types) * scale_prior])

@jit
def log_prior(theta):
    n = int(len(theta) / 2)
    radii, scales = theta[:n], theta[n:]
    return np.sum(norm.logpdf(radii - radius_prior)) + np.sum(norm.logpdf(scales - scale_prior))


def get_predictions(theta, types):
    N = int(len(theta) / 2)
    radii_, scaling_factors_ = theta[:N], theta[N:]

    radii = [radii_[types[i]] for i in range(len(types))]
    scaling_factors = [scaling_factors_[types[i]] for i in range(len(types))]
    return np.array([predict_solvation_free_energy_jax(radii[i], scaling_factors[i], distance_matrices[i], charges[i]) for i in range(len(charges))])


def norm_logpdf(x, mu, sigma):
    """Replacing jax.scipy.stats.norm.logpdf...."""
    return - (x - mu)**2 / (2 * sigma**2) - np.log(np.sqrt(2 * np.pi * sigma **2))

def log_likelihood_of_predictions(predictions):
    if ll == 'student-t':
        log_likelihood_value = np.sum(student_t.logpdf(predictions - expt_means, scale=expt_uncs, df=7))
    elif ll == 'gaussian':
        log_likelihood_value = np.sum(norm_logpdf(x=predictions, mu=expt_means, sigma=expt_uncs))
    else:
        raise (RuntimeError('invalid ll value'))
    return log_likelihood_value

def log_prob(theta, types):
    p = log_prior(theta)
    l = log_likelihood_of_predictions(get_predictions(theta, types))
    return p + l


from jax import grad

if __name__ == '__main__':
    from time import time


    def f(theta):
        return log_prob(theta, dummy_types)

    # VALUE

    print('computing value of log_prob for first time...')
    t0 = time()
    _ = f(dummy_theta)
    t1 = time()
    print('...that took {:.4}s'.format(t1 - t0))

    print('computing value of log_prob a second time...')
    t0 = time()
    _ = f(dummy_theta)
    t1 = time()
    print('...that took {:.4}s'.format(t1 - t0))


    # GRADIENT

    print('computing gradient of log_prob for first time...')
    t0 = time()
    _ = grad(f)(dummy_theta)
    t1 = time()
    print('...that took {:.4}s'.format(t1 - t0))

    print('computing gradient of log_prob a second time...')
    t0 = time()
    p = grad(f)(dummy_theta)
    t1 = time()
    print('...that took {:.4}s'.format(t1 - t0))

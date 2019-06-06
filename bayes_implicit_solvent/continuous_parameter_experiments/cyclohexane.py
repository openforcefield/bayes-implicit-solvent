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
n_conf = 25

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/short_run/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)
np.random.seed(0)
np.random.shuffle(paths_to_samples)

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
mol_names = ['ethane', 'benzene', 'cyclopentane', 'ethylene', 'methane', 'cyclopropane', 'cyclohexane']


for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning], ll=ll)

    if mol.mol_name in mol_names:
        print(mol.mol_name)
        mols.append(mol)



import numpy as np
element_inds = []
element_dict = {'C': 0, 'H': 1}
for mol in mols:
    element_inds.append(np.array([element_dict[a.element.symbol] for a in list(mol.top.atoms())]))

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp


def fast_predict_factory(distance_matrices, charges, element_ind_array):

    @jit
    def predict_solvation_free_energy_jax(theta):
        radii_, scaling_factors_ = theta[:2], theta[2:]

        radii = radii_[element_ind_array]
        scaling_factors = scaling_factors_[element_ind_array]

        @jit
        def compute_component(distance_matrix):
            return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

        W_F = vmap(compute_component)(distance_matrices)

        w_F = W_F * kj_mol_to_kT
        return one_sided_exp(w_F)
    _ = predict_solvation_free_energy_jax(np.ones(4))
    return predict_solvation_free_energy_jax

from tqdm import tqdm
print('jit-compiling things...')
fast_predictors = [fast_predict_factory(mol.distance_matrices, mol.charges, element_inds[i]) for i, mol in enumerate(mols)]

def get_predictions(theta):
    return np.array([f(theta) for f in fast_predictors])


expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncs = np.array([mol.experimental_uncertainty for mol in mols])

from scipy.stats import t as student_t
from scipy.stats import norm

def log_likelihood(predictions):
    #return np.sum(norm.logpdf(predictions, loc=expt_means, scale=expt_uncs))
    return np.sum(student_t.logpdf(predictions, loc=expt_means,
                     scale=expt_uncs,
                     df=7))


prior_location = np.array([0.17, 0.12, 0.72, 0.85]) # mbondi2 set
if __name__ == '__main__':

    def log_likelihood_of_params(theta):
        predictions = get_predictions(theta)
        return log_likelihood(predictions)


    from bayes_implicit_solvent.samplers import random_walk_mh
    x0 = prior_location

    def log_prob_fun(theta):
        if (min(theta) < 0.01) or (max(theta) > 5):
            return -np.inf
        else:
            return np.sum(norm.logpdf(theta - prior_location)) + log_likelihood_of_params(theta)
    mh_result = random_walk_mh(x0, log_prob_fun, n_steps=50000, stepsize=0.025)

    np.savez('cyclohexane_and_friends.npz',
             traj=mh_result,
             expt_means=expt_means,
             expt_uncs=expt_uncs,
             )

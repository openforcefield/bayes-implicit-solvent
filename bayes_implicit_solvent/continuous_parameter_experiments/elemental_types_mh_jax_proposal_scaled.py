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
#ll = 'gaussian'  # or 'student-t'
ll = 'student-t'
randomize_theta0 = False
n_conf = 50

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


import numpy as np
element_inds = []
all_elements = ['S', 'Cl', 'F', 'C', 'I', 'N', 'Br', 'H', 'P', 'O']
N = len(all_elements)
element_dict = dict(zip(all_elements, range(len(all_elements))))

initial_radius_dict = dict(H=0.12, C=0.17, N=0.155, O=0.15, F=0.15,
                   P=0.185, S=0.18, Cl=0.17, Br=0.15, I=0.15)
initial_scaling_factor_dict = dict(H=0.85, C=0.72, N=0.79, O=0.85, F=0.88,
                           P=0.86, S=0.96, Cl=0.80, Br=0.80, I=0.80)


for mol in mols:
    element_inds.append(np.array([element_dict[a.element.symbol] for a in list(mol.top.atoms())]))

from jax import jit, vmap
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp


#def fast_predict_factory(distance_matrices, charges, element_ind_array):

@jit
def predict_solvation_free_energy_jax(theta, distance_matrices, charges, element_ind_array):
    radii_, scaling_factors_ = theta[:N], theta[N:]

    radii = radii_[element_ind_array]
    scaling_factors = scaling_factors_[element_ind_array]

    @jit
    def compute_component(distance_matrix):
        return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, charges)

    W_F = vmap(compute_component)(distance_matrices)

    w_F = W_F * kj_mol_to_kT
    return one_sided_exp(w_F)
#_ = predict_solvation_free_energy_jax(np.ones(N * 2))
#return predict_solvation_free_energy_jax

def get_predictions(theta):
    return np.array([predict_solvation_free_energy_jax(theta, mol.distance_matrices, mol.charges, element_inds[i]) for (i, mol) in enumerate(mols)])


expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncs = np.array([mol.experimental_uncertainty for mol in mols])

from scipy.stats import t as student_t
from scipy.stats import norm

def log_likelihood(predictions):
    #return np.sum(norm.logpdf(predictions, loc=expt_means, scale=expt_uncs))
    return np.sum(student_t.logpdf(predictions, loc=expt_means,
                     scale=expt_uncs,
                     df=5))


initial_radius_array = [initial_radius_dict[a] for a in all_elements]
initial_scaling_factor_array = [initial_scaling_factor_dict[a] for a in all_elements]
prior_location = np.array(initial_radius_array + initial_scaling_factor_array) # mbondi2 set, except not differentiation H from HN...
#prior_location = np.array([0.17, 0.12, 0.72, 0.85]) # mbondi2 set

from simtk import unit
from bayes_implicit_solvent.constants import beta
def unreduce(value):
    """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
    return value / (beta * unit.kilocalorie_per_mole)

def rmse_in_kcal_mol(x, y):
    x_, y_ = unreduce(x), unreduce(y)
    squared_errors = (x_ - y_)**2
    mean_squared_errors = np.mean(squared_errors)
    root_mean_squared_errors = np.sqrt(mean_squared_errors)
    return root_mean_squared_errors

if __name__ == '__main__':

    def log_likelihood_of_params(theta):
        predictions = get_predictions(theta)
        return log_likelihood(predictions)


    from bayes_implicit_solvent.samplers import random_walk_mh
    x0 = prior_location
    #np.random.seed(5)
    #x0[:N] += np.random.randn(N) * 0.02
    #x0[N:] += np.random.randn(N) * 0.1

    # x0 = prior_location
    # np.random.seed(5)
    # x0[:N] += np.random.randn(N) * 0.01
    # x0[N:] += np.random.randn(N) * 0.05

    #x0 = np.zeros(len(prior_location))
    #np.random.seed(0)
    #x0[:N] = (np.random.rand(N) * (0.5 - 0.01)) + 0.01
    #x0[N:] = (np.random.rand(N) * (1.5 - 0.5)) + 0.5


    def log_prob_fun(theta):
        if (min(theta) < 0.01) or (max(theta) > 5):
            return -np.inf
        else:
            return np.sum(norm.logpdf(theta - prior_location)) + log_likelihood_of_params(theta)
    stepsize = np.ones(len(x0))
    stepsize[:N] = 0.0005
    stepsize[N:] = 0.001
    trajs = []
    log_prob_trajs = []

    predictions = get_predictions(x0)
    #print('RMSE: {} kcal/mol'.format(rmse_in_kcal_mol(predictions, expt_means)))

    from tqdm import tqdm
    trange = tqdm(range(1000))
    for t in trange:
        mh_result = random_walk_mh(x0, log_prob_fun, n_steps=50, stepsize=stepsize, progress_bar=False)
        trajs.append(mh_result[0])
        log_prob_trajs.append(mh_result[1])

        predictions = get_predictions(mh_result[0][-1])
        rmse =rmse_in_kcal_mol(predictions, expt_means)
        trange.set_postfix({'train RMSE': '{:.3f} kcal/mol'.format(rmse), 'log_prob': mh_result[1][-1], 'accept_fraction': mh_result[2]})

        x0 = trajs[-1][-1]

        if t % 20 == 0:
            traj = np.vstack(trajs)
            log_prob_traj = np.hstack(log_prob_trajs)
            x0 = traj[-1]
            np.savez('freesolv_mh_jax_df=5.npz',
                     traj=traj,
                     log_prob_traj=log_prob_traj,
                     expt_means=expt_means,
                     expt_uncs=expt_uncs,
                     cids=cids,
                     elements=all_elements,
                     )
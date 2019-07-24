from glob import glob

from autograd import numpy as np
from pkg_resources import resource_filename
from simtk import unit

from bayes_implicit_solvent.constants import beta
from bayes_implicit_solvent.freesolv import cid_to_smiles
from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model
from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.utils import npy_sample_path_to_unitted_snapshots

# 0. Load all molecules
data_path = resource_filename('bayes_implicit_solvent',
                              'data')
n_configuration_samples = 25
path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))
print('first few CIDs', cids[:5])

molecules = []


def unreduce(value):
    """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
    return value / (beta * unit.kilocalorie_per_mole)


for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = npy_sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning])
    if (unreduce(mol.experimental_value) > -15) and (unreduce(mol.experimental_value) < 5):
        molecules.append(mol)
    else:
        print('discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles,
                                                                                                              cid))

# arrays of unitless (in kT) experimental means and uncertainties
expt_means = np.array([mol.experimental_value for mol in molecules])
expt_uncertainties = np.array([mol.experimental_uncertainty for mol in molecules])

# 2. Define a likelihood function, including "type-assignment"

# write an integer in range(0, n_types) on each atom indicating which type it was assigned
n_types = mbondi_model.number_of_nodes
mbondi_type_slices = [mbondi_model.apply_to_molecule(mol.mol) for mol in molecules]


def pack(radii, scales):
    """Given two length-n_types arrays, concatenate them into a single vector"""
    assert (len(radii) == len(scales))
    n = len(radii)
    assert (n == n_types)
    theta = np.hstack([radii, scales])
    return theta


def unpack(theta):
    """Given a flat parameter array theta of length 2 * n_types,
    unpack it into two arrays of length n_types,
    corresponding to radii and scales"""
    n = int(len(theta) / 2)
    assert (n == n_types)
    radii, scales = theta[:n], theta[n:]
    return radii, scales


def assign_radii_and_scales(theta, type_slices):
    """Assign a radius and scale to each atom, using the types assigned in type_slices

    Parameters
    ----------
    theta : array of length 2 n_types
    type_slices : list of variable-length integer arrays taking values in range(0, n_types)

    Returns
    -------
    parameterized_list : list of length-2 tuples, containing two numpy arrays
    """
    radii, scales = unpack(theta)

    parameterized_list = []
    for i in range(len(molecules)):
        parameterized_list.append((radii[type_slices[i]], scales[type_slices[i]]))
    return parameterized_list


def generate_all_predictions(theta):
    parameterized_list = assign_radii_and_scales(theta, mbondi_type_slices)
    predictions = np.array([molecules[i].predict_solvation_free_energy_autograd(radii, scales) for i, (radii, scales) in
                            enumerate(parameterized_list)])
    return predictions


def compute_likelihood(theta, batch_inds):
    pass


def compute_likelihood_and_gradient(theta, batch_inds):
    pass


# 1. Generate k-fold CV splits deterministically
N_folds = 10
inds = np.arange(len(molecules))
np.random.seed(0)
np.random.shuffle(inds)
fold_size = round(len(molecules) / N_folds)
folds = [inds[fold_size * i:fold_size * (i + 1)] for i in range(N_folds)]


def train_test_split(i):
    assert (i in range(N_folds))
    test_inds = folds[i]
    train_inds = np.hstack(list(folds[:i]) + list(folds[i + 1:]))
    return train_inds, test_inds


# 2. Write a function for producing all predictions


## internal tests
def test_train_test_splitter():
    for i in range(N_folds):
        train_inds, test_inds = train_test_split(i)

        # check training set size is right
        assert (len(train_inds) == (N_folds - 1) * fold_size)

        # check test set size is right
        assert (len(test_inds) == fold_size)

        # check that train set has no overlap with test set
        assert (len(set(train_inds).intersection(set(test_inds))) == 0)

        # check that that the union of train and test set = whole set
        assert (set(train_inds).union(test_inds) == set(inds))


def test_generate_all_predictions():
    """Spot-check that the predicted free energies are reasonable
    (right shape, finite, RMSE (in kcal/mol) in a reasonable range)"""
    theta = np.ones(2 * n_types)
    from time import time
    print('predicting hydration free energies with theta=ones...')
    t0 = time()
    predictions = generate_all_predictions(theta)
    t1 = time()
    print('that took {:.3f} s'.format(t1 - t0))

    # sanity check that the predictions are finite and that there's the right number of them
    assert (len(predictions) == len(molecules))
    assert (np.isfinite(predictions).all())

    # sanity check that the numerical values of the predictions aren't grossly wrong...
    pred_kcal_mol = unreduce(predictions)
    expt_kcal_mol = unreduce(expt_means)

    rmse = np.sqrt(np.mean((pred_kcal_mol - expt_kcal_mol) ** 2))
    print('RMSE for theta all ones: {:.3f} kcal/mol'.format(rmse))

    # first time I ran this test, it was ~7 kcal/mol
    assert (rmse < 10.)
    assert (rmse > 2.)

    # check that running with mbondi radii and scales gives an RMSE closer to like 2.5 kcal/mol
    theta = pack(radii=mbondi_model.get_radii(), scales=mbondi_model.get_scale_factors())

    print('predicting hydration free energies with theta from mbondi model...')
    t0 = time()
    predictions = generate_all_predictions(theta)
    t1 = time()
    print('that took {:.3f} s'.format(t1 - t0))

    # sanity check that the predictions are finite and that there's the right number of them
    assert (len(predictions) == len(molecules))
    assert (np.isfinite(predictions).all())

    # sanity check that the numerical values of the predictions aren't grossly wrong...
    pred_kcal_mol = unreduce(predictions)
    expt_kcal_mol = unreduce(expt_means)

    rmse = np.sqrt(np.mean((pred_kcal_mol - expt_kcal_mol) ** 2))

    # I think it's around 2.4 kcal/mol...
    assert (rmse > 2.)
    assert (rmse < 3.)

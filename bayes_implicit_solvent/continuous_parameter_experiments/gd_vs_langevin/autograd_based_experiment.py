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
n_configuration_samples = 25
allowed_DeltaG_range = (-15, 5)
path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')


def unreduce(value):
    """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
    return value / (beta * unit.kilocalorie_per_mole)


def load_dataset(path_to_vacuum_samples, allowed_DeltaG_range=(-15, 5), n_configuration_samples=5):
    paths_to_samples = glob(path_to_vacuum_samples)

    def extract_cid_key(path):
        i = path.find('mobley_')
        j = path.find('.npy')
        return path[i:j]

    molecules = []
    for path in paths_to_samples:
        cid = extract_cid_key(path)
        smiles = cid_to_smiles[cid]
        vacuum_samples = npy_sample_path_to_unitted_snapshots(path)
        thinning = int(len(vacuum_samples) / n_configuration_samples)
        mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning])

        min_DeltaG, max_DeltaG = allowed_DeltaG_range
        if (unreduce(mol.experimental_value) >= min_DeltaG) and (unreduce(mol.experimental_value) <= max_DeltaG):
            molecules.append(mol)
        else:
            print(
                'discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles,
                                                                                                                cid))
    return molecules


molecules = load_dataset(path_to_vacuum_samples, allowed_DeltaG_range, n_configuration_samples)

# arrays of unitless (in kT) experimental means and uncertainties
expt_means = np.array([mol.experimental_value for mol in molecules])
expt_uncertainties = np.array([mol.experimental_uncertainty for mol in molecules])

# 2. Define a likelihood function, including "type-assignment"

# write an integer in range(0, n_types) on each atom indicating which type it was assigned
n_types = mbondi_model.number_of_nodes
mbondi_type_slices = [mbondi_model.apply_to_molecule(mol.mol) for mol in molecules]

print('saving freesolv_inputs.pkl...')
distance_matrices = [m.distance_matrices for m in molecules]
freesolv_inputs = dict(
    distance_matrices=distance_matrices,
    expt_means=expt_means,
    expt_uncertainties=expt_uncertainties,
    type_slices=mbondi_type_slices,
    charges=[m.charges for m in molecules],
    n_types=n_types,
)
from pickle import dump
with open('freesolv_inputs.pkl', 'wb') as f:
    dump(freesolv_inputs, f)
print('... done!')

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


from autograd.scipy.stats import norm
from autograd.scipy.stats import t as student_t

from bayes_implicit_solvent.constants import min_r, max_r, min_scale, max_scale


def flat_log_prior(theta):
    radii, scales = unpack(theta)
    if (min(radii) < min_r) or (max(radii) > max_r) or (min(scales) < min_scale) or (max(scales) > max_scale):
        return - np.inf
    return (- np.log(max_r - min_r) * len(radii)) + (-np.log(max_scale - min_scale) * len(scales))


def gaussian_log_prior(theta):
    """Maybe put a prior that's Gaussian near mbondi parameters..."""
    raise (NotImplementedError)


def gaussian_log_likelihood(predictions, expt_means, expt_uncertainties):
    components = norm.logpdf(predictions, loc=expt_means, scale=expt_uncertainties)
    return np.sum(components)


def student_t_log_likelihood(predictions, expt_means, expt_uncertainties):
    components = student_t.logpdf(predictions, loc=expt_means, scale=expt_uncertainties, df=7)
    return np.sum(components)


def gaussian_log_likelihood_with_model_error(predictions, expt_means, expt_uncertainties, model_uncertainty):
    components = norm.logpdf(predictions, loc=expt_means, scale=(expt_uncertainties + model_uncertainty))
    return np.sum(components)


# 1. Generate k-fold CV splits deterministically
N_folds = 10
inds = np.arange(len(molecules))
np.random.seed(0)
np.random.shuffle(inds)
fold_size = round(len(molecules) / N_folds)
folds = [inds[fold_size * i:fold_size * (i + 1)] for i in range(N_folds)]

from functools import lru_cache


@lru_cache(maxsize=N_folds)
def train_test_split(i):
    assert (i in range(N_folds))
    test_inds = folds[i]
    train_inds = np.array(list(set(inds).difference(set(test_inds))))
    return train_inds, test_inds

for i in range(N_folds):
    train_inds, test_inds = train_test_split(i)



ll_dict = dict(gaussian=gaussian_log_likelihood, student_t=student_t_log_likelihood)

def log_posterior(theta, fold_index=0, ll='gaussian'):
    log_prior_ = flat_log_prior(theta)  # TODO: don't hardcode things like this... want to be flexible!!

    if log_prior_ > -np.inf:
        predictions = generate_all_predictions(theta)
        train_inds, test_inds = train_test_split(fold_index)

        log_likelihood_ = ll_dict[ll](predictions[train_inds], expt_means[train_inds],
                                                  expt_uncertainties[train_inds])
        return log_prior_ + log_likelihood_
    else:
        print(RuntimeWarning('parameters out of bounds!'))
        return log_prior_


from autograd import grad


def grad_log_posterior(theta, fold_index=0, ll='gaussian'):
    return grad(lambda x: log_posterior(x, fold_index, ll))(theta)


from tqdm import tqdm


def gradient_descent(jac, x0, stepsize, num_iters, callback=None):
    """

    Parameters
    ----------
    x0
    jac : callable, with signature jac(x, i)
    stepsize
    n_steps
    callback: callable, with signature callback(x, i, g)

    Returns
    -------

    """
    traj = [x0]
    for i in tqdm(range(num_iters)):
        g = jac(traj[-1], i)
        traj.append(traj[-1] - stepsize * g)
        if not (callback is None):
            callback(traj[-1], i, g)
    return traj

from autograd.misc.optimizers import adam

optimizer_dict = dict(gradient_descent=gradient_descent, adam=adam)

# TODO:
# * Randomized initial points
# * Vanilla gradient descent
# * Line-search


def run_optimizer(optimizer, theta0, num_iters, cv_fold=0, ll='gaussian'):
    """optimizer has signature optimizer(grad, x0, num_iters, callback)
        where grad has signature (x,i) and callback has signature (x,i,g)
    """


    from time import time
    print("MAP-optimizin'...")
    t0 = time()

    # def loss(theta):
    #    log_posterior_ = log_posterior(theta)
    #    print('fxn evaluation:', theta, log_posterior_)
    #    return - log_posterior_

    def grad_loss(theta, i):
        """Needs to have signature (x, i), where i is the iteration count..."""
        grad_log_posterior_ = grad_log_posterior(theta, fold_index=cv_fold, ll=ll)
        print('grad evaluation:', theta, grad_log_posterior_)
        print('gradient norm: {:.3f}'.format(np.linalg.norm(grad_log_posterior_)))
        return - grad_log_posterior_

    traj = [theta0]
    prediction_traj = [generate_all_predictions(theta0)]

    def callback(x, i, g):
        predictions = generate_all_predictions(x)
        traj.append((x, i, g))
        prediction_traj.append(predictions)

        pred_kcal_mol = unreduce(predictions)
        expt_kcal_mol = unreduce(expt_means)

        rmse = np.sqrt(np.mean((pred_kcal_mol - expt_kcal_mol) ** 2))
        print('RMSE for theta_{}: {:.3f} kcal/mol'.format(i, rmse))

    traj = optimizer(grad_loss, theta0, num_iters=num_iters, callback=callback)

    t1 = time()
    print('that took {:.3f} s'.format(t1 - t0))

    return traj, prediction_traj




from scipy.optimize import minimize


def test_LBFGS_minimization():
    theta0 = np.ones(2 * n_types)
    theta0[:n_types] = 0.1

    from time import time
    print("MAP-optimizin' with L-BFGS-B starting from theta=ones...")
    t0 = time()
    function_evaluations = []
    gradient_evaluations = []

    def loss(theta):
        log_posterior_ = log_posterior(theta)
        # global function_evaluations
        function_evaluations.append((theta, log_posterior_))
        print('fxn evaluation:', len(function_evaluations), theta, log_posterior_)
        return - log_posterior_

    def grad_loss(theta):
        grad_log_posterior_ = grad_log_posterior(theta)
        # global gradient_evaluations
        gradient_evaluations.append((theta, grad_log_posterior_))
        print('grad evaluation:', len(gradient_evaluations), theta, grad_log_posterior_)
        return -grad_log_posterior_

    result = minimize(
        x0=theta0,
        method='L-BFGS-B',
        fun=loss,
        jac=grad_loss,
        bounds=[(min_r, max_r)] * n_types + [(min_scale, max_scale)] * n_types,
        # callback=callback,
        options=dict(disp=True, maxiter=10, maxcor=1)
    )
    t1 = time()
    print('that took {:.3f} s'.format(t1 - t0))

    from pickle import dump
    with open('L-BFGS-B-traj.pkl', 'wb') as f:
        dump(dict(result=result, function_evaluations=function_evaluations, gradient_evaluations=gradient_evaluations),
             f)


## internal tests
def test_train_test_splitter():
    for i in range(N_folds):
        train_inds, test_inds = train_test_split(i)

        # check training set size is approximately right
        assert (abs(len(train_inds) - ((N_folds - 1) * fold_size)) <= 2)

        # check test set size is right
        assert (abs(len(test_inds) - fold_size) <= 2)

        # check that train set has no overlap with test set
        assert (len(set(train_inds).intersection(set(test_inds))) == 0)

        # check that that the union of train and test set = whole set
        assert (set(train_inds).union(test_inds) == set(inds))


def rmse(predictions, inds=inds):
    pred_kcal_mol = unreduce(predictions[inds])
    expt_kcal_mol = unreduce(expt_means[inds])

    rmse = np.sqrt(np.mean((pred_kcal_mol - expt_kcal_mol) ** 2))
    return rmse

def train_test_rmse(predictions, split=0):
    train_inds, test_inds = train_test_split(split)
    train_rmse = rmse(predictions, train_inds)
    test_rmse = rmse(predictions, test_inds)
    return train_rmse, test_rmse

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

    # first time I ran this test, it was ~7.154 kcal/mol
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
    print('RMSE for mbondi model: {:.3f} kcal/mol'.format(rmse))

    # I think it's around 2.4 kcal/mol, but this test is saying something like 2.628 kcal/mol
    assert (rmse > 2.)
    assert (rmse < 3.)
np.random.seed(0)
theta0 = pack(radii=mbondi_model.get_radii(), scales=mbondi_model.get_scale_factors())
n_start = 10
x0s = [(np.random.rand(len(theta0)) * 0.05) - 0.025 + theta0 for _ in range(n_start)]



from collections import namedtuple
Experiment = namedtuple('Experiment', ['x0', 'cv_fold', 'll', 'stepsize'])
gd_stepsize = dict(gaussian=1e-8, student_t=1e-6)
experiments = []
for x0 in x0s:
    for cv_fold in range(N_folds):
        for ll in ll_dict:
            experiments.append(Experiment(x0=x0, cv_fold=cv_fold, ll=ll, stepsize=gd_stepsize[ll]))
print('# experiments: ', len(experiments))

def save(experiment, traj, prediction_traj, name):
    from pickle import dump
    with open(name, 'wb') as f:
        dump(dict(experiment=experiment, traj=traj, prediction_traj=prediction_traj), f)

def run_experiment(experiment, num_iters=5000):
    theta0 = experiment.x0
    cv_fold = experiment.cv_fold
    ll = experiment.ll
    stepsize = experiment.stepsize


    optimizer = lambda g, x0, num_iters, callback: gradient_descent(jac=g, x0=x0, stepsize=stepsize,
                                                                    num_iters=num_iters, callback=callback)

    traj, prediction_traj = run_optimizer(optimizer, theta0=theta0, num_iters=num_iters, cv_fold=cv_fold, ll=ll)

    name = 'll={},k={},hash(theta0)={}.pkl'.format(ll, cv_fold, hash(tuple(theta0)))
    print('saving result to ', name)
    save(experiment, traj, prediction_traj, name)

def save_expt_dataset():
    """The ordering depends on the ordering of the vacuum_samples files!!"""
    import numpy as onp
    onp.savez('expt_dataset.npz', smiles=[m.smiles for m in molecules], expt_means=expt_means,
              expt_uncertainties=expt_uncertainties)

if __name__ == "__main__":
    import sys
    try:
        job_id = int(sys.argv[1]) - 1
    except:
        from time import time
        np.random.seed(int(str(hash(time()))[:5]))
        job_id = np.random.randint(len(experiments))
    print('job_id={}'.format(job_id))
    experiment = experiments[job_id]
    print('experiment: ', experiment)

    run_experiment(experiment)

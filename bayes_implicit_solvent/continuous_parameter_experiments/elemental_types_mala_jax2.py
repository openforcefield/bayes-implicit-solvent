# DOESN'T WORK YET


# Currently gets errors like:

"""
Traceback (most recent call last):
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3267, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-85af1f3b28dd>", line 1, in <module>
    runfile('/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments/elemental_types_mala_jax2.py', wdir='/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments')
  File "/Applications/PyCharm.app/Contents/helpers/pydev/_pydev_bundle/pydev_umd.py", line 197, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "/Applications/PyCharm.app/Contents/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments/elemental_types_mala_jax2.py", line 148, in <module>
    mala_result = MALA(x0, log_prob_fun, grad(log_prob_fun), n_steps=1000, stepsize=0.001)
  File "/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/samplers.py", line 232, in MALA
    grads = [grad_log_prob_fun(traj[-1])]
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/api.py", line 206, in grad_f
    ans, g = value_and_grad_f(*args, **kwargs)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/api.py", line 241, in value_and_grad_f
    ans, vjp_py = vjp(f_partial, *dyn_args)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/api.py", line 563, in vjp
    out_primal, out_vjp = ad.vjp(jaxtree_fun, primals_flat)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/interpreters/ad.py", line 76, in vjp
    out_primal, pval, jaxpr, consts = linearize(traceable, *primals)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/interpreters/ad.py", line 69, in linearize
    jaxpr, out_pval, consts = pe.trace_to_jaxpr(jvpfun, in_pvals)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/interpreters/partial_eval.py", line 314, in trace_to_jaxpr
    jaxpr, (out_pval, consts, env) = fun.call_wrapped(pvals, **kwargs)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/linear_util.py", line 86, in call_wrapped
    ans = self.f(*args, **self.kwargs)
  File "/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments/elemental_types_mala_jax2.py", line 146, in log_prob_fun
    return np.sum(norm.logpdf(theta - prior_location)) + log_likelihood_of_params(theta)# - penalty
  File "/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments/elemental_types_mala_jax2.py", line 137, in log_likelihood_of_params
    return log_likelihood(predictions)
  File "/Users/joshuafass/Documents/GitHub/bayes-implicit-solvent/bayes_implicit_solvent/continuous_parameter_experiments/elemental_types_mala_jax2.py", line 121, in log_likelihood
    return np.sum(norm.logpdf(predictions, loc=expt_means, scale=expt_uncs))
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/scipy/stats/norm.py", line 28, in logpdf
    x, loc, scale = _promote_args_like(osp_stats.norm.logpdf, x, loc, scale)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/numpy/lax_numpy.py", line 192, in _promote_args_like
    return _promote_shapes(*_promote_to_result_dtype(op, *args))
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/numpy/lax_numpy.py", line 164, in _promote_to_result_dtype
    to_dtype = _result_dtype(op, *args)
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/jax/numpy/lax_numpy.py", line 171, in _result_dtype
    return _dtype(op(*args))
  File "/Users/joshuafass/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py", line 1700, in logpdf
    putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
"""

# UPDATE MARCH 10: Runs now, if I replace norm.logpdf with raw numpy...
# However, it hits NaNs immediately, and never accepts any proposals

# UPDATE: It works, but it's way slower than it should be...
# UPDATE: When I include jit compilation, it's super super slow to get started... Like, tens of minutes...


from jax import numpy as np
from numpy import load, random, savez
from bayes_implicit_solvent.molecule import Molecule
from simtk import unit


def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = load(path_to_npy_samples)
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

distance_matrices = [mol.distance_matrices for mol in mols]
charges = [mol.charges for mol in mols]

#TODO: vmap and JIT this...
@jit
def get_predictions(theta):
    return np.array([predict_solvation_free_energy_jax(theta, distance_matrices[i], charges[i], element_inds[i]) for i in range(len(mols))])


expt_means = np.array([mol.experimental_value for mol in mols])
expt_uncs = np.array([mol.experimental_uncertainty for mol in mols])

from scipy.stats import t as student_t
from jax.scipy.stats import norm

@jit
def log_likelihood(predictions):
    return np.sum(-((predictions - expt_means)**2 / (expt_uncs**2)))
    #return np.sum(norm.logpdf(predictions, loc=expt_means, scale=expt_uncs))
    #return np.sum(student_t.logpdf(predictions, loc=expt_means,
    #                 scale=expt_uncs,
    #                 df=7))


initial_radius_array = [initial_radius_dict[a] for a in all_elements]
initial_scaling_factor_array = [initial_scaling_factor_dict[a] for a in all_elements]
prior_location = np.array(initial_radius_array + initial_scaling_factor_array) # mbondi2 set, except not differentiation H from HN...
#prior_location = np.array([0.17, 0.12, 0.72, 0.85]) # mbondi2 set
from jax import grad

@jit
def log_prior(theta):
    return - np.sum(theta**2)

if __name__ == '__main__':

    @jit
    def log_likelihood_of_params(theta):
        predictions = get_predictions(theta)
        return log_likelihood(predictions)


    from bayes_implicit_solvent.samplers import random_walk_mh, MALA, langevin
    x0 = np.array(prior_location)
    v0 = random.randn(len(prior_location)) * 0.01

    @jit
    def log_prob_fun(theta):
        #penalty = min(0.0, 0.01 - np.min(theta))**2
        #penalty += min(0.0, np.max(theta) - 5.0)**2
        # return np.sum(norm.logpdf(theta - prior_location)) + log_likelihood_of_params(theta)# - penalty
        return log_prior(theta) + log_likelihood_of_params(theta)


    #mh_result = random_walk_mh(x0, log_prob_fun, n_steps=100000, stepsize=0.001)
    #mala_result = MALA(x0, log_prob_fun, grad(log_prob_fun), n_steps=1000, stepsize=0.00001)

    #traj, log_prob_traj, grads, acceptance_probs, stepsizes = mala_result
    # np.savez('freesolv_mala_jax.npz',
    #          traj=traj,
    #          log_prob_traj=log_prob_traj,
    #          acceptance_probs=acceptance_probs,
    #          stepsizes=stepsizes,
    #          expt_means=expt_means,
    #          expt_uncs=expt_uncs,
    #          cids=cids,
    #          elements=all_elements,
    #          )

    stepsize = 0.0001
    collision_rate = 1/stepsize

    langevin_result = langevin(x0, v0, log_prob_fun,
                               grad(log_prob_fun),
                               stepsize=stepsize,
                               collision_rate=collision_rate,
                               n_steps=1000)
    traj, log_probs = langevin_result
    savez('freesolv_langevin_jax_big_collision_rate_march13.npz',
             traj=traj,
             log_prob_traj=log_probs,
             expt_means=expt_means,
             expt_uncs=expt_uncs,
             cids=cids,
             elements=all_elements,
             )

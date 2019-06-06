# use metropolis-adjusted langevin to sample the radii and scales in
# the elemental-types-only model, initialized from OBC2 paramaters


import os.path
from autograd import numpy as np
from simtk import unit
from bayes_implicit_solvent.samplers import MALA
from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

from bayes_implicit_solvent.continuous_parameter_experiments.elemental_types_mh import log_prior, mols, ll, data_path, smiles, construct_arrays
smiles_list = smiles

elements = []
charges = []
distance_matrices = []
expt_means = []
expt_uncs = []
vacuum_trajs = []

n_configuration_samples = 25

name = 'n_config={}_{}_ll'.format(n_configuration_samples, ll)
smiles_subset_fname = os.path.join(data_path,
                                   'smiles_subset_{}.txt'.format(name))
with open(smiles_subset_fname, 'w') as f:
    f.writelines(['{}\n'.format(smiles) for smiles in smiles_list])

from bayes_implicit_solvent.utils import get_charges
from scipy.spatial.distance import pdist, squareform

for mol in mols:
    expt_means.append(mol.experimental_value)
    expt_uncs.append(mol.experimental_uncertainty)

    elements.append(np.array([a.element.atomic_number for a in mol.top.atoms()]))
    charges.append(get_charges(mol.sys))
    distance_matrices.append([squareform(pdist(snapshot / unit.nanometer)) for snapshot in mol.vacuum_traj])

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


from bayes_implicit_solvent.gb_models.numpy_gb_models import compute_OBC_energy_vectorized


def log_prob(theta):
    L = log_prior(theta)
    if not (L > -np.inf):
        return L
    else:
        parameterized_list = construct_arrays(theta)

        for i in range(len(mols)):
            radii, scales = parameterized_list[i]
            W_F = np.array([compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges[i]) for distance_matrix in
                            distance_matrices[i]])
            w_F = W_F * kj_mol_to_kT
            pred_free_energy = one_sided_exp(w_F)

            if ll == 'gaussian':
                L += norm.logpdf(pred_free_energy, loc=expt_means[i], scale=expt_uncs[i] ** 2)
            else:
                L += student_t.logpdf(pred_free_energy, loc=expt_means[i],
                                         scale=expt_uncs[i] ** 2,
                                         df=7)
    return L

if __name__ == '__main__':
    n_types = mbondi_model.number_of_nodes
    print('n_types: {}'.format(n_types))

    initial_radii = np.array(mbondi_model.get_radii())
    initial_scales = np.array(mbondi_model.get_scale_factors())
    theta0 = np.hstack((initial_radii, initial_scales))

    # print('initial theta', theta0)
    # initial_log_prob = parallel_log_prob(theta0)
    # print('initial log prob', log_prob(theta0))
    #
    # #grad_log_prob = grad(log_prob)
    # grad_log_prob = parallel_grad_log_prob
    #
    # print('initial gradient norm = {}'.format(np.linalg.norm(grad_log_prob(theta0))))

    # minimized_theta_fname = os.path.join(data_path,
    #                                      'elemental_types_l-bfgs_freesolv_{}.npy'.format(
    #                                          name))
    #
    # print('minimizing...')
    # from scipy.optimize import minimize
    #
    # loss = lambda theta: -parallel_log_prob(theta)
    # grad_loss = lambda theta: -parallel_grad_log_prob(theta)
    # bounds = [(0.001, 2.0)] * len(theta0[:-1]) + [(-np.inf, np.inf)]
    # result = minimize(loss, theta0,
    #               jac=grad(loss),
    #               method='L-BFGS-B',
    #               bounds=bounds,
    #               options={'disp': True,
    #                        'maxiter': 5})

    # theta1 = result.x
    # np.save(minimized_theta_fname, theta1)
    # print('theta after initial minimization', theta1)
    # print('gradient norm after initial minimization = {}'.format(np.linalg.norm(grad_log_prob(theta1))))

    # 5. Run MALA

    stepsize = 1e-8
    n_steps = 2000
    traj, log_probs, grads, acceptance_probabilities, stepsizes = MALA(theta0, log_prob, grad(log_prob),
                                                                       n_steps=n_steps, stepsize=stepsize,
                                                                       adapt_stepsize=True)

    np.savez(os.path.join(data_path,
                          'elemental_types_mala_freesolv_{}.npz'.format(
                              name)),
             traj=traj, grads=grads, acceptance_probabilities=acceptance_probabilities, stepsizes=stepsizes)

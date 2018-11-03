# use metropolis-adjusted langevin to sample the radii and scales in
# the elemental-types-only model


import os.path

import mdtraj as md
from pkg_resources import resource_filename
from autograd import numpy as np
from bayes_implicit_solvent.posterior_sampling import Molecule
from bayes_implicit_solvent.samplers import MALA
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots
from simtk import unit
data_path = '../data/'
gaussian_ll = True

if __name__ == '__main__':
    import sys

    try:
        job_id = int(sys.argv[1])
        print("Using supplied job_id: ", job_id)
    except:
        job_id = np.random.randint(1000)
        print("No valid job_id supplied! Selecting one at random: ", job_id)


    # 1. load the dataset, precompute distance matrices
    np.random.seed(job_id)

    inds = np.arange(len(smiles_list))
    np.random.shuffle(inds)
    inds = inds[:int(len(smiles_list) * 0.75)]
    smiles_subset = [smiles_list[i] for i in inds]

    mols = []
    elements = []
    charges = []
    distance_matrices = []
    expt_means = []
    expt_uncs = []

    n_configuration_samples = 10 # TODO: Since this is cheaper, can probably modify this a bit...

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
    from bayes_implicit_solvent.solvation_free_energy import kB, temperature

    kj_mol_to_kT = 1.0 * unit.kilojoule_per_mole / (kB * temperature)

    def one_sided_exp(w_F):
        DeltaF = - (logsumexp(- w_F) - np.log(len(w_F)))
        return DeltaF

    # TODO: parallelize using multiprocessing

    all_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    element_dict = dict(zip(all_elements, list(range(len(all_elements)))))

    def pack(radii, scales):
        n = len(radii)
        theta = np.zeros(2 * n)
        theta[:n] = radii
        theta[n:2*n] = scales
        return theta

    def unpack(theta):
        n = int((len(theta) ) / 2)
        radii, scales = theta[:n], theta[n:2*n]
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

    from bayes_implicit_solvent.numpy_gb_models import compute_OBC_energy_vectorized


    #from multiprocessing import Pool
    #n_processes = 4
    #pool = Pool(n_processes)
    #def log_prob_component(i, theta):
    #    radii, scales = construct_array(i, theta)
    #    if min(np.min(radii), np.min(scales)) < 0.001:
    #        print('out of bounds!')
    #        return -np.inf
    #    W_F = np.array([compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges[i]) for distance_matrix in
    #                    distance_matrices[i]])
    #    w_F = W_F * kj_mol_to_kT
    #    pred_free_energy = one_sided_exp(w_F)
    #    #return student_t.logpdf(pred_free_energy, loc=expt_means[i],
    #    #                scale=expt_uncs[i]**2,
    #    #                df=7)
    #    return norm.logpdf(pred_free_energy, loc=expt_means[i], scale=expt_uncs[i] ** 2)

    def log_prob(theta):
        mol_radii, mol_scales = construct_arrays(theta)
        if min(theta[:-1]) < 0.001 or max(theta[:-1]) > 2:
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
                                         scale=expt_uncs[i] ** 2,
                                         df=7)
        return logp
    #def log_prob(theta):
    #    return sum(map(lambda i: log_prob_component(i, theta), range(len(mols))))

    #def parallel_log_prob(theta):
    #    return sum(pool.map(lambda i: log_prob_component(i, theta), range(len(mols))))

    #def parallel_grad_log_prob(theta):
    #    """Use autograd to compute the gradient of each term separately, then return the sum"""
    #    return sum(pool.map(lambda i: grad(log_prob_component(i, theta), argnum=1)(theta), range(len(mols))))

    # 3. Take gradient of likelihood function

    # 4. Minimize for a few steps
    initial_radii = np.ones(len(all_elements)) * 0.12
    initial_scales = np.ones(len(all_elements)) * 0.85

    theta0 = pack(initial_radii, initial_scales)
    print('initial theta', theta0)
    initial_log_prob = log_prob(theta0)
    print('initial log prob', log_prob(theta0))

    grad_log_prob = grad(log_prob)

    print('initial gradient norm = {}'.format(np.linalg.norm(grad_log_prob(theta0))))

    minimized_theta_fname = os.path.join(data_path,
                         'elemental_types_l-bfgs_freesolv_{}.npy'.format(
                             name))

    print('minimizing...')
    from scipy.optimize import minimize
    loss = lambda theta: -log_prob(theta)
    bounds = [(0.001, 2.0)] * len(theta0[:-1]) + [(-np.inf, np.inf)]
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

    stepsize = 1e-8
    n_steps = 2000
    traj, log_probs, grads, acceptance_probabilities, stepsizes = MALA(theta1, log_prob, grad_log_prob, n_steps=n_steps, stepsize=stepsize, adapt_stepsize=True, )

    np.savez(os.path.join(data_path,
                         'elemental_types_mala_freesolv_{}.npz'.format(
                             name)),
             traj=traj, grads=grads, acceptance_probabilities=acceptance_probabilities, stepsizes=stepsizes)

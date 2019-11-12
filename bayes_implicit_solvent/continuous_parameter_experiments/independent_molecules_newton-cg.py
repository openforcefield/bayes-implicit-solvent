# use unadjusted langevin to sample the radii and scales in
# the every-atom-for-itself model


import os.path

import mdtraj as md
from autograd import numpy as np
from pkg_resources import resource_filename
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'
gaussian_ll = True
randomize_theta0 = False


for i in range(len(smiles_list)):
    print('{} of {}'.format(i, len(smiles_list)))
#for i in range(2):

    smiles_subset = [smiles_list[i]]

    mols = []
    elements = []
    charges = []
    distance_matrices = []
    expt_means = []
    expt_uncs = []
    vacuum_trajs = []

    n_configuration_samples = 100  # TODO: Since this is cheaper, can probably modify this a bit...

    name = 'n_config={}_smiles_ind={}'.format(n_configuration_samples, i)
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

        #mol_radii = np.array([radii[element_dict[element]] for element in elements[i]])
        #mol_scales = np.array([scales[element_dict[element]] for element in elements[i]])
        #return mol_radii, mol_scales

        return radii, scales


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
        assert(len(mols) == 1)
        n_types = mols[0].n_atoms
        #n_types = len(set(element_dict.values()))


        # 4. Minimize for a few steps
        min_theta = 0.001
        max_theta = 2.0

        initial_radii = np.ones(n_types) * 0.12
        initial_scales = np.ones(n_types) * 0.85

        theta0 = pack(initial_radii, initial_scales)
        if randomize_theta0:
            theta0 += 0.01 * np.random.randn(len(theta0))
            theta0[theta0 < min_theta] = min_theta
            theta0[theta0 > max_theta] = max_theta

        print('initial theta', theta0)
        initial_log_prob = log_prob(theta0)
        print('initial log prob', initial_log_prob)
        print('initial gradient of log prob', grad_log_prob(theta0))

        print('initial gradient norm = {}'.format(np.linalg.norm(grad_log_prob(theta0))))

        minimized_theta_fname = os.path.join(data_path,
                                             'all_types_newton-cg_freesolv_{}.npy'.format(
                                                 name))

        print('minimizing...')
        from scipy.optimize import minimize
        from autograd import hessian_vector_product
        loss = lambda theta: - log_prob(theta)
        bounds = [(min_theta, max_theta)] * len(theta0)
        result = minimize(loss, theta0,
                      jac=grad(loss),
                      #method='L-BFGS-B',
                      method='Newton-CG',
                      hessp=hessian_vector_product(loss),
                      bounds=bounds,
                      options={'disp': True,
                               'maxiter': 100})

        theta1 = result.x
        np.save(minimized_theta_fname, theta1)
        print('theta after initial minimization', theta1)
        print('gradient norm after minimization = {}'.format(np.linalg.norm(grad_log_prob(theta1))))

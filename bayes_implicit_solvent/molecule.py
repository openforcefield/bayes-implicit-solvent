import numpy as np
from scipy.stats import t as student_t
from simtk import unit

from bayes_implicit_solvent.constants import min_r, max_r, min_scale, max_scale
from bayes_implicit_solvent.solvation_free_energy import predict_solvation_free_energy, \
    get_vacuum_samples, create_implicit_sim, beta
from bayes_implicit_solvent.freesolv import db, smiles_list, mol_top_sys_pos_list


import jax.numpy as np
from bayes_implicit_solvent.gb_models.jax_gb_models import compute_OBC_energy_vectorized

from jax import grad, jit, vmap
from bayes_implicit_solvent.solvation_free_energy import kj_mol_to_kT, one_sided_exp

from scipy.spatial.distance import pdist, squareform
from bayes_implicit_solvent.utils import get_charges

# TODO: Maybe resurrect this...
# from functools import lru_cache

class Molecule():
    def __init__(self, smiles, verbose=False, vacuum_samples=None,
                 n_samples=50, thinning=50000, ll='gaussian'):
        """Create an object that supports prediction of solvation free energy given radii and scaling factors

        Parameters
        ----------
        smiles : string
            SMILES representation of molecule

        verbose : boolean
            whether to print updates

        vacuum_samples : list of unit'd snapshots

        n_samples : int
            if vacuum_samples not provided, how many samples to collect?

        thinning : int
            if vacuum samples not provided, how many steps to take between samples?

        ll : string in {'gaussian', 'student-t'}
            is each log-likelihood term Gaussian or heavy-tailed?

        Attributes
        ----------
        mol_index_in_smiles_list : int
            where in the sorted smiles list are we
            TODO: Replace with less janky indexing
        mol_index_in_freesolv : int
            where in freesolv are we
            TODO: Replace with less janky indexing
        mol_name : string
            iupac name (or alternative if IUPAC is unavailable or not parseable by OEChem
        mol : OEMol object
            OEMol object with explicit hydrogens, partial charges, etc. assigned
        top : OpenMM Topology
        sys : OpenMM System
        pos : [n_atoms x 3] array
            initial atomic positions
        atom_names : list of strings
            element symbol + occurrence number, e.g. ['C1', 'F1', 'F2', 'F3', 'Br1']
        n_atoms : int
            number of atoms
        vacuum_sim : OpenMM Simulation
            simulation object at vacuum using Smirnoff parameters, Reference platform, BAOAB Langevin integrator
        vacuum_traj : [n_snapshots x n_atoms x 3] array
            collection of atom positions sampled by vacuum_sim
        experimental_value : float
            experimental value of solvation free energy for this molecule, converted from kcal/mol to unitless (in kT)
        experimental_uncertainty
            uncertainty in experimental_value, converted from kcal.mol to unitless (in kT)
        implicit_sim : OpenMM Simulation
            same as vacuum_sim, but with a GBSAOBCForce added
        """

        assert (ll in {'gaussian', 'student-t'})
        self.smiles = smiles
        self.verbose = verbose
        self.ll = ll

        # find the index of this molecule in our smiles list
        mol_index_in_smiles_list = -1
        for i in range(len(smiles_list)):
            if smiles_list[i] == smiles:
                mol_index_in_smiles_list = i
        self.mol_index_in_smiles_list = mol_index_in_smiles_list
        if self.mol_index_in_smiles_list == -1:
            raise (ValueError(
                "the smiles string queried ({}) doesn't appear to be in FreeSolv's SMILES list...".format(smiles)))

        # find the index of this molecule in Freesolv
        mol_index_in_freesolv = -1
        for i in range(len(db)):
            if db[i][1] == smiles:
                mol_index_in_freesolv = i
        self.mol_index_in_freesolv = mol_index_in_freesolv
        if self.mol_index_in_freesolv == -1:
            raise (ValueError("the smiles string queried ({}) doesn't appear to be in FreeSolv...".format(smiles)))

        self.mol_name = db[mol_index_in_freesolv][2]

        self.mol, self.top, self.sys, self.pos = mol_top_sys_pos_list[self.mol_index_in_smiles_list]
        self.atom_names = [a.name for a in self.top.atoms()]
        self.n_atoms = len(self.pos)

        self._n_samples = n_samples
        self._thinning = thinning
        self.charges = get_charges(self.sys)

        if type(vacuum_samples) == type(None):
            if verbose: print('collecting vacuum samples...')
            self.vacuum_sim, self.vacuum_traj = get_vacuum_samples(self.top, self.sys, self.pos,
                                                                   n_samples=self._n_samples,
                                                                   thinning=self._thinning)
        else:
            self.vacuum_traj = vacuum_samples
            self._n_samples = len(vacuum_samples)

        self.distance_matrices = np.array([squareform(pdist(snapshot / unit.nanometer)) for snapshot in self.vacuum_traj])

        # both in reduced units
        self.experimental_value = beta * (float(db[mol_index_in_freesolv][3]) * unit.kilocalorie_per_mole)
        self.experimental_uncertainty = beta * (float(db[mol_index_in_freesolv][4]) * unit.kilocalorie_per_mole)

        if verbose:
            print('creating implicit-solvent simulation...')
        self.implicit_sim = create_implicit_sim(self.top, self.sys)

        if verbose:
            print('successfully initialized {}'.format(self.mol_name))

    def predict_solvation_free_energy(self, radii, scaling_factors):
        """Use one-sided EXP to predict the solvation free energy using this set of radii

        Parameters
        ----------
        scaling_factors
        radii : array of floats, either unit'd or assumed to be in nanometers
            radius parameters for each atom
        scaling_factors : array of floats
            scalingFactors for each atom

        Returns
        -------
        mean : float
        uncertainty : float
        """
        assert (len(radii) == self.n_atoms)
        assert (len(scaling_factors) == self.n_atoms)
        return predict_solvation_free_energy(self.implicit_sim, self.vacuum_traj, radii, scaling_factors)


    #@jit
    def predict_solvation_free_energy_jax(self, radii, scaling_factors):
        @jit
        def compute_component(distance_matrix):
            return compute_OBC_energy_vectorized(distance_matrix, radii, scaling_factors, self.charges)

        W_F = vmap(compute_component)(self.distance_matrices)

        w_F = W_F * kj_mol_to_kT
        return one_sided_exp(w_F)

    def gaussian_log_likelihood(self, radii, scaling_factors):
        """Un-normalized log-likelihood using Gaussian located at experimentally measured value, with
        scale set by the estimated experimental error.

        This will be sensitive to the stated experimental uncertainty, which we are somewhat skeptical of,
        since for many (most?) entries in FreeSolv this just a default value.


        N(mean_sim | mu = mean_expt,
                     sigma^2 = sigma_expt * max(sigma_expt, sigma_sim)
        """
        simulation_mean, simulation_uncertainty = self.predict_solvation_free_energy(radii, scaling_factors)

        mu = self.experimental_value
        sigma2 = self.experimental_uncertainty * max([self.experimental_uncertainty, simulation_uncertainty])

        return - (simulation_mean - mu) ** 2 / sigma2

    def log_likelihood(self, radii, scaling_factors):
        """To be more robust to inaccurate statement of `experimental_uncertainty`, favor a Student-t likelihood
        over a Gaussian likelihood. This also corresponds to using a nuisance parameter for the experimental uncertainty (with
        an inverse-Gamma prior?) and marginalizing it out.

        TODO: Gelman reference
        """
        simulation_mean, simulation_uncertainty = self.predict_solvation_free_energy(radii, scaling_factors)

        mu = self.experimental_value
        sigma = np.sqrt(self.experimental_uncertainty * max([self.experimental_uncertainty, simulation_uncertainty]))

        return student_t.logpdf(simulation_mean, loc=mu,
                                scale=sigma,  # TODO: Look up how best to put scale information here
                                df=7)  # TODO: Decide what to use for the degrees of freedom parameter

    def bounds_check(self, radii, scale_factors):
        """Check whether parameters are in bounds (radii in (max_r, min_r),
        scale_factors in (min_scale, max_scale)
        """
        # TODO: refactor to have a parameter object that knows how to check its own bounds
        if (np.min(radii) < min_r) or (np.max(radii) > max_r) or \
                (np.min(scale_factors) < min_scale) or (np.max(scale_factors) > max_scale):
            return False
        else:
            return True

    def log_prob_uncached(self, radii, scale_factors):
        """Un-normalized log-probability : log-likelihood, if log-prior > -infty
        """
        if self.bounds_check(radii, scale_factors):
            if self.ll == 'student-t':
                ll = self.log_likelihood(radii, scale_factors)
            elif self.ll == 'gaussian':
                ll = self.gaussian_log_likelihood(radii, scale_factors)
            return ll
        else:
            return - np.inf


    # TODO: Return also a list of all the predictions

    # @lru_cache(maxsize=4)
    # TODO: Maybe use lru_cache again
    def log_prob(self, radii, scale_factors):
        return self.log_prob_uncached(radii, scale_factors)


if __name__ == '__main__':
    from bayes_implicit_solvent.samplers import random_walk_mh

    np.random.seed(0)

    smiles = 'C'
    mol = Molecule(smiles, n_samples=10, thinning=1000)
    n = 2
    radii0 = np.ones(n)
    scaling_factors0 = np.ones(n)
    theta0 = np.hstack([radii0, scaling_factors0])
    assert (len(theta0) == 2 * n)


    def unpack(theta):
        _radii = theta[:n]
        radii = np.zeros(mol.n_atoms)
        radii[0] = _radii[0]
        radii[1:] = _radii[1]

        _scaling_factors = theta[n:]
        scaling_factors = np.zeros(mol.n_atoms)
        scaling_factors[0] = _scaling_factors[0]
        scaling_factors[1:] = _scaling_factors[1]

        return radii, scaling_factors


    def L(theta):
        radii, scaling_factors = unpack(theta)
        return mol.log_prob(radii, scaling_factors)


    traj, log_probs, acceptance_fraction = random_walk_mh(theta0, L,
                                                          n_steps=1000000, stepsize=0.02)
    import os.path

    data_path = 'data/'
    np.save(os.path.join(data_path, 'radii_samples_{}.npy'.format(smiles)), traj[:, :n])
    np.save(os.path.join(data_path, 'scale_samples_{}.npy'.format(smiles)), traj[:, n:])
    np.save(os.path.join(data_path, 'theta_samples_{}.npy'.format(smiles)), traj)
    np.save(os.path.join(data_path, 'log_probs_{}.npy'.format(smiles)), log_probs)

    print(acceptance_fraction)
    print('atom_names: ', mol.atom_names)

    posterior_predictions = [mol.predict_solvation_free_energy(*unpack(theta))[0] for theta in traj[::100]]

    np.save(os.path.join(data_path, 'posterior_predictions_{}.npy'.format(smiles)), posterior_predictions)

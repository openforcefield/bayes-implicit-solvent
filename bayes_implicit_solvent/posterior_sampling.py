import numpy as np
from scipy.stats import t as student_t
from simtk import unit

from bayes_implicit_solvent.solvation_free_energy import predict_solvation_free_energy, \
    get_vacuum_samples, db, smiles_list, mol_top_sys_pos_list, create_implicit_sim, beta


class Molecule():
    def __init__(self, smiles, verbose=False, vacuum_samples=None):
        """Create an object that supports prediction of solvation free energy given radii

        Parameters
        ----------
        smiles : string
            SMILES representation of molecule

        verbose : boolean
            whether to print updates

        vacuum_samples : list of unit'd snapshots

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
        self.smiles = smiles
        self.verbose = verbose

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

        if type(vacuum_samples) == type(None):
            if verbose: print('collecting vacuum samples...')
            # TODO: maybe expose these parameters, if we need this not to be hard-coded...
            self._n_samples = 50
            self._thinning = 50000
            self.vacuum_sim, self.vacuum_traj = get_vacuum_samples(self.top, self.sys, self.pos,
                                                                   n_samples=self._n_samples,
                                                                   thinning=self._thinning)
        else:
            # self.vacuum_sim, _ = get_vacuum_samples(self.top, self.sys, self.pos, n_samples=2, thinning=2)
            self.vacuum_traj = vacuum_samples

        # both in reduced units
        self.experimental_value = beta * (float(db[mol_index_in_freesolv][3]) * unit.kilocalorie_per_mole)
        self.experimental_uncertainty = beta * (float(db[mol_index_in_freesolv][4]) * unit.kilocalorie_per_mole)

        if verbose: print('creating implicit-solvent simulation...')
        self.implicit_sim = create_implicit_sim(self.top, self.sys)

        if verbose: print('successfully initialized {}'.format(self.mol_name))

    def predict_solvation_free_energy(self, radii):
        """Use one-sided EXP to predict the solvation free energy using this set of radii

        Parameters
        ----------
        radii : array of floats, either unit'd or assumed to be in nanometers
            radius parameters for each atom

        Returns
        -------
        mean : float
        uncertainty : float
        """
        assert (len(radii) == self.n_atoms)
        return predict_solvation_free_energy(self.implicit_sim, radii, self.vacuum_traj)

    def gaussian_log_likelihood(self, radii):
        """Un-normalized log-likelihood using Gaussian located at experimentally measured value, with
        scale set by the estimated experimental error.

        This will be sensitive to the stated experimental uncertainty, which we are somewhat skeptical of,
        since for many (most?) entries in FreeSolv this just a default value.


        N(mean_sim | mu = mean_expt,
                     sigma^2 = sigma_expt * max(sigma_expt, sigma_sim)


        """
        simulation_mean, simulation_uncertainty = self.predict_solvation_free_energy(radii)

        mu = self.experimental_value
        sigma2 = self.experimental_uncertainty * max([self.experimental_uncertainty, simulation_uncertainty])

        return - (simulation_mean - mu) ** 2 / sigma2

    def log_likelihood(self, radii):
        """To be more robust to inaccurate statement of `experimental_uncertainty`, favor a Student-t likelihood
        over a Gaussian likelihood. This also corresponds to using a nuisance parameter for the experimental uncertainty (with
        an inverse-Gamma prior?) and marginalizing it out.

        TODO: Gelman reference
        """
        simulation_mean, simulation_uncertainty = self.predict_solvation_free_energy(radii)

        mu = self.experimental_value
        sigma = np.sqrt(self.experimental_uncertainty * max([self.experimental_uncertainty, simulation_uncertainty]))

        return student_t.logpdf(simulation_mean, loc=mu,
                                scale=sigma, # TODO: Look up how best to put scale information here
                                df=7) # TODO: Decide what to use for the degrees of freedom parameter

    def log_prior(self, radii):
        """Un-normalized log-prior: uniform in [0.01, 1.0]^n_atoms"""
        min_r, max_r = 0.01, 1.0
        dim = len(radii)
        if (np.min(radii) < min_r) or (np.max(radii) > max_r):
            return - np.inf
        else:
            return dim * np.log(max_r - min_r)  # uniform prior, normalized

    def log_prob(self, radii):
        """Un-normalized log-probability : log-prior + log-likelihood"""
        prior = self.log_prior(radii)
        if prior > -np.inf:
            #ll = self.log_likelihood(radii)  # TODO: Switch back to Student-t?
            ll = self.gaussian_log_likelihood(radii)
            return prior + ll
        else:
            return prior


if __name__ == '__main__':
    from bayes_implicit_solvent.samplers import random_walk_mh

    np.random.seed(0)

    smiles = 'C'
    mol = Molecule(smiles)
    radii0 = np.ones(len(mol.pos))

    traj, log_probs, acceptance_fraction = random_walk_mh(radii0, mol.log_prob,
                                                          n_steps=100000, stepsize=0.01)
    import os.path

    data_path = 'data/'
    np.save(os.path.join(data_path, 'radii_samples_{}.npy'.format(smiles)), traj)

    print(acceptance_fraction)
    print('atom_names: ', mol.atom_names)

    # posterior_predictions = [mol.predict_solvation_free_energy(r)[0] for r in traj[::100]]

from collections import defaultdict
import numpy as np
from simtk import unit

from simtk import unit

temperature = 298.15 * unit.kelvin  # conditions from FreeSolv
pressure = 1.01325 * unit.bar  # conditions from FreeSolv
from openmmtools.constants import kB

kT = kB * temperature
from openeye import oechem

# 0. Load the systems and coordinates precomputed for FreeSolv
print('loading pickles')
from pickle import load
with open('pickles/some_generated_systems.pkl', 'rb') as f:
    generated_systems = load(f) # list of (mol, topology, system, positions) tuples
print(len(generated_systems))

# TODO: must regenerate these, since the default GBSA parameters I used were very wrong probably...
with open('pickles/some_generated_samples.pkl', 'rb') as f:
    generated_samples = load(f) # list of { "xyz_vacuum": xyz_vacuum, "xyz_implicit": xyz_implicit } dicts

def load_freesolv(path="../../FreeSolv-0.51/database.txt"):
    """Loads the freesolv database as a list of lists of strings"""
    with open(path, 'r') as f:
        freesolv = f.read()

    legend = freesolv.split('\n')[2].split('; ')
    db = []
    for entry in freesolv.split('\n')[3:-1]:
        db.append(entry.split('; '))
    return legend, db

legend, db = load_freesolv()
expt_measurements = [(float(entry[3]), float(entry[4])) for entry in db]

# 1. Construct models, one for each forcefield
from openforcefield.typing.engines.smirnoff import ForceField


def get_parameter_ids(ff, mol):
    """Get a list with the SMIRNOFF parameter id of each atom in the input OEMol object"""

    # pass in a length-1 list, receive a length-1 list, assign its item to "all_labels"
    all_labels = ff.labelMolecules([mol])[0]

    # "all_labels" is a dict, with keys:
    #    ['HarmonicBondGenerator', 'HarmonicAngleGenerator', 'PeriodicTorsionGenerator', 'NonbondedGenerator']
    nonbonded_labels = all_labels['NonbondedGenerator']

    # "nonbonded_labels" is a list of 3-tuples,
    # where the 2nd element in each tuple is a string like 'n3' or 'n15' parameter id assigned
    parameter_ids = [label[1] for label in nonbonded_labels]
    return parameter_ids

def parameter_id_to_index(parameter_id):
    """Converts a SMIRNOFF parameter id string (like 'n3' or 'n15') to an 0-start integer index"""

    assert(parameter_id[0] == 'n') # make sure this is a nonbonded parameter...

    return int(parameter_id[1:]) - 1

from glob import glob
forcefield_paths = glob('forcefields/*.ffxml')
# make sure we only keep the modified forcefields
forcefield_paths = [path for path in forcefield_paths if '_' in path]

active_gbsa_param_sets = defaultdict(set)

print('applying each forcefield to each molecule')
from tqdm import tqdm
for forcefield in tqdm(forcefield_paths[:5]):

    truncation = int(forcefield.split('.')[0].split('_')[-1])

    ff = ForceField(forcefield)

    for i in range(len(generated_systems)):
        mol = generated_systems[i][0]
        active_gbsa_param_sets[truncation].update(get_parameter_ids(ff, mol))

for key in sorted(active_gbsa_param_sets.keys()):
    print(key, len(active_gbsa_param_sets[key]))


# let's select a single model, the one with just two atom types, so we can visualize them

n_types = 10
gbsa_forcefield = ForceField([path for path in forcefield_paths if '_{}.ffxml'.format(n_types) in path][0])

def generate_initial_params(n_types):
    return np.hstack([[1.5]*n_types, [0.5]*n_types])
initial_params = generate_initial_params(n_types)

# note, gbsa_forcefield will only be used for the gbsa radius definitions

from openforcefield.typing.engines.smirnoff.gbsaforces import OBC2

from multiprocessing import Pool
n_subprocesses = 4

kT = kB * temperature

from openmmtools.integrators import BAOABIntegrator
from simtk import unit
from simtk import openmm as mm
from simtk.openmm import app
import numpy as np
from tqdm import tqdm


def generate_sim(topology, system, platform_name="Reference"):
    if platform_name == "Reference":
        platform = mm.Platform.getPlatformByName("Reference")
        platform_properties = None
    elif platform_name == "CUDA":
        platform = mm.Platform.getPlatformByName("CUDA")
        platform_properties = {"CudaPrecision": "mixed"}

    baoab = BAOABIntegrator(collision_rate=1 / unit.picosecond, timestep=2 * unit.femtosecond, temperature=temperature)
    sim = app.Simulation(topology, system, baoab, platform=platform, platformProperties=platform_properties)
    return sim

def generate_implicit_system_from_vacuum_system(system, topology):
    from copy import deepcopy
    implicit_system = deepcopy(system)

    #attempt 1: find the OBC force, and put it on the implicit system
    #dummy_forcefield = app.ForceField('amber99_obc.xml')
    #dummy_system = dummy_forcefield.createSystem(topology)
    # TODO: Document why this didn't work

    #attempt 2: create implicit solvent system directly
    #dummy_system = topology.createSystem(implicitSolvent=OBC2)
    # TODO: Document why this didn't work

    #attempt 3: add each particle manually
    atom_symbols = [a.element.symbol for a in topology.atoms()]

    # (radius, scale)
    default = (1.5, 0.5)
    initial_params = {
        'H': (1.2, 1.425952),
        'C': (1.7, 1.058554),
        'N': (1.55, 0.733599),
        'O': (1.5, 1.061039),
        'F': (1.5, 0.500000),
        'Si': (2.1, 0.500000),
        'P': (1.85, 0.500000),
        'S': (1.8 - 0.703469, 0.867814),
        'Cl': (1.7, 0.500000),
        'Br': (1.5, 0.500000),
        'I': (1.5, 0.500000),
    }

    nb_force = [f for f in system.getForces() if 'Nonbonded' in f.__class__.__name__][0]

    obc2 = OBC2()
    for i in range(topology.getNumAtoms()):
        # get (radius, scale) from defaults
        if atom_symbols[i] in initial_params:
            radius, scale = initial_params[atom_symbols[i]]
        else:
            radius, scale = default

        # get charge from whatever was assigned in nonbonded force
        charge = nb_force.getParticleParameters(i)[0]

        # (charge (elementary charge), radius (nm), scale (nm))
        obc2.addParticle((charge, radius, scale))
    implicit_system.addForce(obc2)

    print([f.__class__.__name__ for f in implicit_system.getForces()])

    return implicit_system


class Molecule():
    """"""
    def __init__(self, mol, top, sys, pos, xyz_vacuum=[], xyz_implicit=[], expt=0, expt_stdev=1, gbsa_forcefield=gbsa_forcefield):
        """

        :param mol: oechem.oemol
        :param top: openmm.topology
        :param sys: openmm.system
        :param pos: (n_atoms,3) coordinate array
        :param xyz_vacuum: (n_samples, n_atoms, 3) coordinates array
        :param xyz_implicit: (n_samples, n_atoms, 3) coordinates array
        :param expt: experimental solvation free energy (kT)
        :param expt_stdev: stdev of experimental solvation free energy (kT)
        :param gbsa_forcefield: forcefield to use for GBSA "types"
        """

        self.mol, self.top, self.sys, self.pos = mol, top, sys, pos
        self.n_atoms = self.top.getNumAtoms()

        self.vacuum_sim, self.implicit_sim = self.generate_sims(top, sys)

        if len(xyz_vacuum) == 0 or len(xyz_implicit) == 0:
            print('generating some samples')
            xyz_vacuum, xyz_implicit = self.get_samples()


        self.all_snapshots = list(xyz_implicit) + list(xyz_vacuum)
        self.mbar = self.mbar_from_endpoints(xyz_implicit, xyz_vacuum)

        self.expt = expt
        self.expt_stdev = expt_stdev

        self.gbsa_forcefield = gbsa_forcefield
        self.type_list = [parameter_id_to_index(id) for id in get_parameter_ids(self.gbsa_forcefield, self.mol)]

    def generate_sims(self, topology, system):
        implicit_system = generate_implicit_system_from_vacuum_system(system, topology)
        # Generate reference-platform simulations we'll use for energy calculations later
        vacuum_sim = generate_sim(topology, system, "Reference")
        implicit_sim = generate_sim(topology, implicit_system, "Reference")
        return vacuum_sim, implicit_sim

    def simulate(self, sim, init_pos, n_snapshots=5, thinning=1000):
        """Collect samples for a given topology / system"""

        sim.context.setPositions(init_pos)
        sim.minimizeEnergy()
        sim.step(thinning)

        snapshots = []
        for _ in tqdm(range(n_snapshots)):
            sim.step(thinning)
            snapshots.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True))

        xyz = np.array([coords.value_in_unit(unit.nanometers) for coords in snapshots])
        return xyz

    def get_samples(self):
        xyz_vacuum = self.simulate(self.vacuum_sim, self.pos)
        xyz_implicit = self.simulate(self.implicit_sim, self.pos)

        return xyz_vacuum, xyz_implicit

    def mbar_from_endpoints(self, xyz_implicit, xyz_vacuum):
        """Given samples (and simulators) in implicit solvent and vacuum, return
        an MBAR object and the solvation free energy..."""
        snapshot_list = [xyz_implicit, xyz_vacuum]
        all_snapshots = list(snapshot_list[0]) + list(snapshot_list[1])
        N_k = [len(snapshots) for snapshots in snapshot_list]

        u_vacuum = []
        u_implicit = []

        for xyz in all_snapshots:
            self.vacuum_sim.context.setPositions(xyz)
            self.implicit_sim.context.setPositions(xyz)
            u_vacuum.append(self.vacuum_sim.context.getState(getEnergy=True).getPotentialEnergy() / kT)
            u_implicit.append(self.implicit_sim.context.getState(getEnergy=True).getPotentialEnergy() / kT)
        u_kn = np.array([u_vacuum, u_implicit])
        # TODO: Do some sort of validation that this is producing
        from pymbar import MBAR

        mbar = MBAR(u_kn, N_k)
        return mbar


    def parse_params(self, params):
        mid = int(len(params) / 2)
        radii_by_type, scales_by_type = params[:mid], params[mid:]

        # need radii_by_type and scales_by_type to be iterable
        if len(params) == 2:
            radii_by_type = [radii_by_type]
            scales_by_type = [scales_by_type]

        radii = np.zeros(self.n_atoms)
        scales = np.zeros(self.n_atoms)
        for i in range(self.n_atoms):
            radii[i] = radii_by_type[self.type_list[i]]
            scales[i] = scales_by_type[self.type_list[i]]

        return radii, scales

    def set_obc2_params(self, params):
        """Set the obc2 radii and scale factors of the implicit solvent sim"""

        obc2 = [f for f in self.implicit_sim.system.getForces() if 'GBForce' in f.__class__.__name__][0]

        radii, scales = self.parse_params(params)

        for i in range(self.n_atoms):
            charge, _, scale = obc2.getParticleParameters(i)
            obc2.setParticleParameters(i, (charge, radii[i], scales[i]))
        obc2.updateParametersInContext(self.implicit_sim.context)

    def get_perturbed_solvation_free_energy(self, params):
        """Use an mbar object to estimate the free energy from radius/scale parameters..."""
        self.set_obc2_params(params)
        u_perturbed = []
        for xyz in self.all_snapshots:
            self.implicit_sim.context.setPositions(xyz)
            u_perturbed.append(self.implicit_sim.context.getState(getEnergy=True).getPotentialEnergy() / kT)
        u_ln = np.vstack((self.mbar.u_kn, u_perturbed))
        return self.mbar.computePerturbedFreeEnergies(u_ln)[0][0, 2]

# generate a list of all the molecules from the database, and generate a few samples each
molecules = []
for i in range(len(generated_systems)):
    mol, top, sys, pos = generated_systems[i]
    samples = generated_samples[i]
    molecules.append(Molecule(mol, top, sys, pos,
                              #samples['xyz_vacuum'], samples['xyz_implicit'], # don't look at generated samples
                              expt=expt_measurements[i][0],
                              expt_stdev=expt_measurements[i][1],
                              gbsa_forcefield=gbsa_forcefield))

def get_pred(mp):
    molecule, params = mp
    return molecule.get_perturbed_solvation_free_energy(params)


def log_likelihood(params):
    """Gaussian likelihood using experimental mean and standard deviation"""
    if np.min(params) <= 0:
        return -np.inf

    ll_components = np.zeros(len(molecules))
    preds = np.zeros(len(molecules))
    #from multiprocessing import Pool
    #pool = Pool(n_subprocesses)
    #preds = pool.map(get_pred, [(molecule, params) for molecule in molecules])

    for i, molecule in enumerate(molecules):
        preds[i] = molecule.get_perturbed_solvation_free_energy(params)
        ll_components[i] = - (preds[i] - molecule.expt) ** 2 / (2 * molecule.expt_stdev ** 2)
    return np.sum(ll_components)

if __name__ == "__main__":
    n_dims = len(initial_params)

    # 3. Sample using emcee
    import emcee
    def run_emcee(lnprobfn):
        nwalkers = n_dims * 2
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                              dim=n_dims,
                              lnpostfn=lnprobfn, threads=n_subprocesses)

        X0 = np.vstack([initial_params + np.random.randn(n_dims)*0.01 for _ in range(nwalkers)])
        sampler.run_mcmc(X0, 1000)

        return sampler

    print('Running MCMC')
    sampler = run_emcee(log_likelihood)

    def save_sampler(sampler, name='sampler_results_{}.npz'.format(n_types)):
        np.savez(name,
                 flatchain=sampler.flatchain,
                 flatlnprobability=sampler.flatlnprobability,
                 acceptance_fraction=sampler.acceptance_fraction,
                 )
    save_sampler(sampler)

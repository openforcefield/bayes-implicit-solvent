import numpy as onp
from autograd import numpy as np
import pymbar
from pkg_resources import resource_filename

from bayes_implicit_solvent.utils import apply_per_particle_params_to_GB_force
from bayes_implicit_solvent.utils import get_gbsa_force, get_nb_force

from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit
from tqdm import tqdm
from openmmtools.integrators import BAOABIntegrator

stepsize = 1.0 * unit.femtosecond
collision_rate = 1.0 / unit.picosecond

from bayes_implicit_solvent.constants import temperature, beta


def construct_gbsa_force(system):
    """Creates a GBSAOBCForce with the same charges as in system.

    Parameters
    ----------
    system : OpenMM System

    Returns
    -------
    gbsa : openmm.GBSAOBCForce
        force with the same particle charges as in system, but radius[i]=scalingFactor[i]=1.0
    """
    nb_force = get_nb_force(system)

    gbsa = mm.GBSAOBCForce()
    for i in range(nb_force.getNumParticles()):
        charge = nb_force.getParticleParameters(i)[0]
        gbsa.addParticle(charge=charge, radius=1.0, scalingFactor=1.0)
    return gbsa


#from autograd.scipy.special import logsumexp
from jax.scipy.special import logsumexp
# from jax import jit
from simtk import unit
from bayes_implicit_solvent.constants import kB, temperature

kj_mol_to_kT = 1.0 * unit.kilojoule_per_mole / (kB * temperature)

#@jit
def one_sided_exp(w_F, compute_uncertainty=False):
    """Clone of pymbar EXP that plays nice with autograd / jax.

    (Cloned from https://github.com/choderalab/pymbar/blob/69d1f0ff680e9ac1c6a51a5a207ea28f3ed86740/pymbar/exp.py#L54-L132 )

    Parameters
    ----------
    w_F : array
        unitless works
    compute_uncertainty : bool
        whether to return an estimate of the uncertainty in DeltaF

    Returns
    -------
    DeltaF : float
        estimated free energy difference
    dDeltaF : float (optional, if compute_uncertainty==True)
        estimated standard deviation of DeltaF estimator
    """
    DeltaF = - (logsumexp(- w_F) - np.log(len(w_F)))
    if compute_uncertainty:
        x = np.exp(- w_F - np.max(-w_F))
        dDeltaF = np.std(x) / (np.mean(x) * np.sqrt(len(x)))
        return DeltaF, dDeltaF
    return DeltaF

def get_vacuum_samples(topology, system, positions, n_samples=100, thinning=10000):
    """Runs a simulation for n_samples * thinning timesteps and returns snapshots.

    Parameters
    ----------
    topology : OpenMM Topology
    system : OpenMM System
    positions : [n_atoms x 3] array of floats, unit'd or assumed to be in nanometers
        initial atomic coordinates
    n_samples : integer
        number of samples to collect
    thinning : integer
        number of integrator steps between saved snapshots

    Returns
    -------
    vacuum_sim : OpenMM Simulation
        simulation constructed from topology, system, BAOABIntegrator, and reference platform
    vacuum_traj : len(n_samples)-list of unit'd [n_atoms x 3] float-arrays
        samples collected at vacuum state
    """
    vacuum_sim = app.Simulation(topology,
                                system,
                                BAOABIntegrator(temperature=temperature,
                                                collision_rate=collision_rate,
                                                timestep=stepsize,
                                                measure_heat=False,
                                                measure_shadow_work=False,
                                                ),
                                platform=mm.Platform.getPlatformByName('Reference')
                                )
    vacuum_sim.context.setPositions(positions)
    vacuum_sim.context.setVelocitiesToTemperature(temperature)
    vacuum_sim.minimizeEnergy()
    vacuum_sim.step(thinning)

    vacuum_traj = []
    for _ in tqdm(range(n_samples)):
        vacuum_sim.step(thinning)
        vacuum_traj.append(vacuum_sim.context.getState(getPositions=True).getPositions(asNumpy=True))

    return vacuum_sim, vacuum_traj


from copy import deepcopy


def create_implicit_sim(topology, system):
    """Creates an OpenMM simulation with an implicit-solvent force in fg0.

    Parameters
    ----------
    topology : OpenMM Topology
    system : OpenMM System
        system to deepcopy and add a GBSAOBCForce to...
        shouldn't include an implicit solvent force already!

    Returns
    -------
    implicit_sim : OpenMM Simulation
        Implicit solvent force will be force group 0, everything else will be force group 1
    """
    gbsa = construct_gbsa_force(system)
    new_system = deepcopy(system)
    new_system.addForce(gbsa)

    implicit_sim = app.Simulation(topology,
                                  new_system,
                                  mm.LangevinIntegrator(temperature,  # note: expected to be unused
                                                        collision_rate,
                                                        stepsize),  # note: expected to be unused
                                  platform=mm.Platform.getPlatformByName('Reference')
                                  )

    for force in implicit_sim.system.getForces():
        force.setForceGroup(1)

    gbsa.setForceGroup(0)

    return implicit_sim


def get_implicit_u_diffs(implicit_sim, samples):
    """Get reduced-potential energy differences for all samples.

    Parameters
    ----------
    implicit_sim : OpenMM Simulation
        simulation object containing the implicit solvent force in force-group 0, and all other forces in other force-groups...

    samples : [n_samples x n_atoms x 3] array of floats, unit'd or assumed to be in nanometers
        assumed to be drawn from equilibrium for vacuum version of implicit_sim

    Returns
    -------
    u_diff : numpy array, dtype=float
        reduced-potential energy differences for all samples
        u_diff[i] = u_implicit(samples[i]) - u_vacuum(samples[i])

    """
    u_diff = np.zeros(len(samples))
    for i in range(len(samples)):
        implicit_sim.context.setPositions(samples[i])
        u_diff[i] = beta * implicit_sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
    return u_diff


def predict_solvation_free_energy(implicit_sim, vacuum_traj, radii, scaling_factors):
    """Apply one-sided EXP to estimate GBSA-predicted solvation free energy at the specified Born radii, given vacuum samples.

    Parameters
    ----------

    implicit_sim : OpenMM Simulation
        simulation object containing the implicit solvent force in force-group 0, and all other forces in other force-groups...
    vacuum_traj : [n_samples x n_atoms x 3] array of floats, unit'd or assumed to be in nanometers
        collection of samples drawn from equilibrium for a system identical to implicit_sim, minus the GBSA force
    radii : array of floats, unit'd or assumed to be in nanometers
        Born radii for all atoms
    scaling_factors : array of floats
        scalingFactors for all atoms


    Returns
    -------
    mean : float, unitless (kT)
        one-sided EXP estimate of solvation free energy
    uncertainty : float, unitless (kT)
        closed-form estimate of uncertainty (stddev) of the one-sided EXP estimate
        # TODO: detect when this is likely an underestimate of the uncertainty, and return bootstrapped estimate instead...
    """
    gb_force = get_gbsa_force(implicit_sim.context.getSystem())
    # TODO: Check occasionally that the force group partition matches expectation?

    apply_per_particle_params_to_GB_force(radii, scaling_factors, gb_force)
    gb_force.updateParametersInContext(implicit_sim.context)
    u_diffs = get_implicit_u_diffs(implicit_sim, vacuum_traj)
    mean, uncertainty = pymbar.EXP(u_diffs)

    return mean, uncertainty

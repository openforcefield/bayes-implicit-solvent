# Generate xyz samples in vacuum and implicit solvent for each molecule...
from openforcefield.typing.engines.smirnoff.gbsaforces import OBC2
from pickle import load
with open('generated_systems.pkl', 'rb') as f:
    generated_systems = load(f)

from simtk import unit

temperature = 298.15 * unit.kelvin  # conditions from FreeSolv
pressure = 1.01325 * unit.bar  # conditions from FreeSolv
from openmmtools.constants import kB

from multiprocessing import Pool
n_subprocesses = 32

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

def simulate(topology, system, initial_positions, n_snapshots=100, thinning=10000, platform_name="Reference"):
    """Collect samples for a given topology / system"""

    sim = generate_sim(topology, system, platform_name)
    sim.context.setPositions(initial_positions)
    sim.minimizeEnergy()
    sim.step(thinning)

    snapshots = []
    for _ in tqdm(range(n_snapshots)):
        sim.step(thinning)
        snapshots.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True))

    xyz = np.array([coords.value_in_unit(unit.nanometers) for coords in snapshots])
    return xyz

def generate_implicit_system_from_vacuum_system(system, topology):
    from copy import deepcopy
    implicit_system = deepcopy(system)
    obc2 = OBC2()
    for i in range(topology.getNumAtoms()):
        obc2.addParticle((1.0, 1.0, 1.0)) # TODO : Fix this to use actual parameters...
    implicit_system.addForce(obc2)
    return implicit_system

generated_samples = []

def get_samples(generated_system):

    (mol, topology, system, positions) = generated_system
    print(topology)

    xyz_vacuum = simulate(topology, system, positions)

    # Collect samples in implicit solvent
    implicit_system = generate_implicit_system_from_vacuum_system(system, topology)
    xyz_implicit = simulate(topology, implicit_system, positions, platform_name="Reference")

    return { "xyz_vacuum": xyz_vacuum, "xyz_implicit": xyz_implicit }

if __name__ == "__main__":
    #generated_samples = []
    #for generated_system in tqdm(generated_systems):
    #    generated_samples.append(get_samples(generated_system))

    pool = Pool(n_subprocesses)
    generated_samples = pool.map(get_samples, generated_systems)

    from pickle import dump
    with open('generated_samples.pkl', 'wb') as f:
        dump(generated_samples, f)

import os.path
import numpy as np
import pymbar

from bayes_implicit_solvent.utils import get_gbsa_force, get_nb_force

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
path_to_freesolv = os.path.join(data_path, 'FreeSolv-0.51/database.txt')

with open(path_to_freesolv, 'r') as f:
    freesolv = f.read()


def load_freesolv():
    db = []
    for entry in freesolv.split('\n')[3:-1]:
        db.append(entry.split('; '))
    return db


db = load_freesolv()

from pickle import load

with open('mol_top_sys_pos.pkl', 'rb') as f:
    mol_top_sys_pos_list = load(f)

with open('sorted_smiles.pkl', 'rb') as f:
    smiles_list = load(f)

from simtk.openmm import app
from simtk import openmm as mm
from simtk import unit
from tqdm import tqdm
from openmmtools.integrators import BAOABIntegrator

from openmmtools.constants import kB

temperature = 298 * unit.kelvin
beta = 1.0 / (kB * temperature)


def construct_gbsa_force(system):
    nb_force = get_nb_force(system)

    gbsa = mm.GBSAOBCForce()
    for i in range(nb_force.getNumParticles()):
        charge = nb_force.getParticleParameters(i)[0]
        gbsa.addParticle(charge=charge, radius=1.0, scalingFactor=1.0)
    return gbsa


def get_vacuum_samples(topology, system, positions, n_samples=100, thinning=10000):
    vacuum_sim = app.Simulation(topology,
                                system,
                                BAOABIntegrator(temperature=temperature,
                                                collision_rate=1.0 / unit.picosecond,
                                                timestep=1.0 * unit.femtosecond,
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


from simtk import unit

from copy import deepcopy


def create_implicit_sim(topology, system):
    """Implicit solvent will be force group 0, everything else will be force group 1"""
    gbsa = construct_gbsa_force(system)
    new_system = deepcopy(system)
    new_system.addForce(gbsa)

    implicit_sim = app.Simulation(topology,
                                  new_system,
                                  mm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                                        1.0 * unit.femtosecond),
                                  platform=mm.Platform.getPlatformByName('Reference')
                                  )

    for force in implicit_sim.system.getForces():
        force.setForceGroup(1)

    gbsa.setForceGroup(0)

    return implicit_sim


def get_implicit_u_diffs(implicit_sim, x):
    u_diff = np.zeros(len(x))
    for i in range(len(x)):
        implicit_sim.context.setPositions(x[i])
        u_diff[i] = beta * implicit_sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
    return u_diff


def initialize_molecule_representation(mol_index_in_smiles_list):
    mol, top, sys, pos = mol_top_sys_pos_list[mol_index_in_smiles_list]
    # atom_names = [a.name for a in top.atoms()]

    # vacuum_sim, vacuum_traj = get_vacuum_samples(top, sys, pos, n_samples=50, thinning=50000)

    deepcopy(sys)

    experimental_value = beta * (float(db[mol_index_in_freesolv][3]) * unit.kilocalorie_per_mole)

    experimental_uncertainty = beta * (float(db[mol_index_in_freesolv][4]) * unit.kilocalorie_per_mole)

    implicit_sim = create_implicit_sim(top, sys)

    return mol, top, sys, pos, experimental_value, experimental_uncertainty, implicit_sim


def predict_solvation_free_energy(implicit_sim, radii, vacuum_traj):
    gbsa = get_gbsa_force(implicit_sim.context.getSystem())

    for i in range(len(radii)):
        charge = gbsa.getParticleParameters(i)[0]
        gbsa.setParticleParameters(index=i,
                                   charge=charge,
                                   radius=radii[i],
                                   scalingFactor=1.0,
                                   )

    gbsa.updateParametersInContext(implicit_sim.context)
    u_diffs = get_implicit_u_diffs(implicit_sim, vacuum_traj)
    mean, unc = pymbar.EXP(u_diffs)

    return mean, unc


if __name__ == '__main__':
    # let's find one of the smallest molecules
    n_atom_list = [len(t[-1]) for t in mol_top_sys_pos_list]

    mol_index = np.argmin(n_atom_list)
    smiles = smiles_list[mol_index]

    smallest_n_atoms = min(n_atom_list)
    mol_indices = [i for i in range(len(n_atom_list)) if n_atom_list[i] == smallest_n_atoms]
    smiles_of_smallest_molecules = [smiles_list[mol_index] for mol_index in mol_indices]
    print(smiles_of_smallest_molecules)

    for n in sorted(list(set(n_atom_list))):
        mol_indices = [i for i in range(len(n_atom_list)) if n_atom_list[i] == n]
        print('# atoms: {}'.format(n))
        print('\t' + str([smiles_list[mol_index] for mol_index in mol_indices]) + '\n')

    # pick one with 5 atoms and 3 distinct elements
    smiles = 'C(F)(F)(F)Br'
    mol_index_in_smiles_list = -1
    for i in range(len(smiles_list)):
        if smiles_list[i] == smiles:
            mol_index_in_smiles_list = i

    mol_index_in_freesolv = -1
    for i in range(len(db)):
        if db[i][1] == smiles:
            mol_index_in_freesolv = i

    print(db[mol_index_in_freesolv])

    mol_name = db[mol_index_in_freesolv][2]
    print('mol_name : ', mol_name)

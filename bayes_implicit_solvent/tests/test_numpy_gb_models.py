# ensure consistency between energies from OpenMM and from autograd / numpy clone

from autograd import numpy as np
from openmmtools.testsystems import AlanineDipeptideImplicit
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app

from bayes_implicit_solvent.gb_models.numpy_gb_models import compute_OBC_energy_reference, compute_OBC_energy_vectorized


# first pass: just set up the system in a function: can later make this conform to pytest fixture idiom...

def set_up_comparison_system():
    testsystem = AlanineDipeptideImplicit()

    gbsa_force = testsystem.system.getForce(4)

    print('COMPUTED VALUES\n\n')
    for i in range(gbsa_force.getNumComputedValues()):
        variable, expression, computation_type = gbsa_force.getComputedValueParameters(i)
        expressions = expression.split(';')
        print('computation type: {}'.format(computation_type))
        print('{} = {}'.format(variable, expressions[0]))
        for e in expressions[1:]:
            print('\t{}'.format(e.strip()))
        print()

    print('ENERGY TERMS\n\n')
    for i in range(gbsa_force.getNumEnergyTerms()):
        expression, computation_type = gbsa_force.getEnergyTermParameters(i)
        expressions = expression.split(';')
        print('computation type: {}'.format(computation_type))
        print(expressions[0])
        for e in expressions[1:]:
            print('\t{}'.format(e.strip()))
        print()

    # quick test that the clone is doing the same as the original
    implicit_sim = app.Simulation(testsystem.topology,
                                  testsystem.system,
                                  mm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                                        1.0 * unit.femtosecond),
                                  platform=mm.Platform.getPlatformByName('CPU')
                                  )
    implicit_sim.context.setPositions(testsystem.positions)
    implicit_sim.step(1)

    from tqdm import tqdm

    print('collecting a few configurations...')
    implicit_traj = []
    for _ in tqdm(range(100)):
        implicit_sim.step(100)
        implicit_traj.append(implicit_sim.context.getState(getPositions=True).getPositions(asNumpy=True))

    from scipy.spatial.distance import pdist, squareform

    offset = 0.009
    screening = 138.935456
    surface_tension = 28.3919551
    solvent_dielectric = 78.5
    solute_dielectric = 1.0

    implicit_only = AlanineDipeptideImplicit()
    implicit_only.system.removeForce(0)
    implicit_only.system.removeForce(0)
    implicit_only.system.removeForce(0)
    implicit_only.system.removeForce(0)
    implicit_only.system.removeForce(1)
    print(implicit_only.system.getForces())

    implicit_only_sim = app.Simulation(testsystem.topology,
                                       implicit_only.system,
                                       mm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                                             1.0 * unit.femtosecond),
                                       platform=mm.Platform.getPlatformByName('CPU')
                                       )
    implicit_only_sim.context.setPositions(testsystem.positions)

    implicit_only_Us = []
    for i in range(len(implicit_traj)):
        implicit_only_sim.context.setPositions(implicit_traj[i])
        implicit_only_Us.append(implicit_only_sim.context.getState(getEnergy=True).getPotentialEnergy())

    print('cloning per particle parameters')

    charges = []
    o_rs = []
    s_rs = []

    for i in range(gbsa_force.getNumPerParticleParameters()):
        print(gbsa_force.getPerParticleParameterName(i))

    for i in range(gbsa_force.getNumParticles()):
        charge, o_r, s_r = gbsa_force.getParticleParameters(i)
        charges.append(charge)
        o_rs.append(o_r)
        s_rs.append(s_r)
    charges = np.array(charges)
    radii = np.zeros(len(o_rs))
    for i in range(len(o_rs)):
        radii[i] = o_rs[i] + offset
    scales = np.zeros(len(s_rs))
    for i in range(len(o_rs)):
        scales[i] = s_rs[i] / o_rs[i]

    def precompute_distance_matrices(x):
        return [squareform(pdist(snapshot)) + np.eye(len(snapshot)) for snapshot in x]

    distance_matrices = precompute_distance_matrices(implicit_traj)

    autograd_reference_energies = [compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                                                offset, screening, surface_tension,
                                                                solvent_dielectric, solute_dielectric) for
                                   distance_matrix in
                                   distance_matrices]

    autograd_vectorized_energies = [compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges,
                                                                  offset, screening, surface_tension,
                                                                  solvent_dielectric, solute_dielectric) for
                                    distance_matrix in
                                    distance_matrices]

    openmm_energies = [v.value_in_unit(unit.kilojoule_per_mole) for v in implicit_only_Us]

    return openmm_energies, autograd_reference_energies, autograd_vectorized_energies


openmm_energies, autograd_reference_energies, autograd_vectorized_energies = set_up_comparison_system()


def plot_comparison(openmm_energies, autograd_energies):
    import matplotlib.pyplot as plt
    from bayes_implicit_solvent.utils import remove_top_right_spines

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.scatter(autograd_energies, openmm_energies)
    plt.xlabel('autograd energies')
    plt.ylabel('openmm energies')
    plt.plot(sorted(autograd_energies), sorted(autograd_energies), color='grey')
    plt.plot(sorted(openmm_energies), sorted(openmm_energies), color='grey')
    remove_top_right_spines(ax)
    plt.savefig('energy-comparison.png', dpi=300)
    plt.close()

from numpy.testing import assert_allclose

rtol = 1e-5


def test_numpy_reference_consistent_with_openmm():
    assert_allclose(autograd_reference_energies, openmm_energies, rtol=rtol)


def test_numpy_reference_consistent_with_numpy_vectorized():
    assert_allclose(autograd_vectorized_energies, autograd_reference_energies, rtol=rtol)

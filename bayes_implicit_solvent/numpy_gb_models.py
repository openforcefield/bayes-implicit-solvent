"""For prototyping, since openmm doesn't support gradients w.r.t. per-particle parameters in CustomGBForce
"""

from autograd import numpy as np

def step(x):
    # return (x > 0)
    return 1.0 * (x >= 0)


# first, implement with for loops: later, vectorize
def compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                 offset=0.009, screening=138.935484, surface_tension=28.3919551,
                                 solvent_dielectric=78.5, solute_dielectric=1.0,
                                 ):
    """Trying to mimic the OpenMM implementation """
    N = len(radii)
    assert ((len(charges) == N) and (distance_matrix.shape == (N, N)))

    # first, compute this intermediate value that depends on pairs
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # compute offset and scaled radii
                or1 = radii[i] - offset
                or2 = radii[j] - offset
                sr2 = scales[j] * or2

                # distance between the two particles
                r = distance_matrix[i, j]

                L = max(or1, abs(r - sr2))
                U = r + sr2
                I[i, j] = step(r + sr2 - or1) * 0.5 * (
                            1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * np.log(
                        L / U) / r)

    I = np.sum(I, axis=1)

    # okay, next compute born radii
    B = np.zeros(N)
    for i in range(N):
        offset_radius = radii[i] - offset
        psi = I[i] * offset_radius

        psi_coefficient = 0.8
        psi2_coefficient = 0
        psi3_coefficient = 2.909125

        psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (psi3_coefficient * psi ** 3)

        B[i] = 1 / (1 / offset_radius - np.tanh(psi_term) / radii[i])

    # finally, compute three energy terms
    E = 0

    # single particle
    for i in range(N):
        E += surface_tension * (radii[i] + 0.14) ** 2 * (radii[i] / B[i]) ** 6
        E += -0.5 * screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges[i] ** 2 / B[i]

    # particle pairs
    for i in range(N):
        for j in range(i):
            r = distance_matrix[i, j]
            f = np.sqrt(r ** 2 + B[i] * B[j] * np.exp(-r ** 2 / (4 * B[i] * B[j])))

            E += -screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges[i] * charges[j] / f

    return E

def compute_OBC_energy_vectorized(distance_matrix, radii, scales, charges,
                                  offset=0.009, screening=138.935484, surface_tension=28.3919551,
                                  solvent_dielectric=78.5, solute_dielectric=1.0,
                                  ):
    """Replacing for-loops with vectorized operations"""
    N = len(radii)
    r = distance_matrix + np.eye(N) # so I don't have divide-by-zero nonsense
    or1 = radii.reshape((N, 1)) - offset
    or2 = radii.reshape((1, N)) - offset
    sr2 = scales.reshape((1, N)) * or2

    L = np.maximum(or1, abs(r - sr2))
    U = r + sr2
    I = step(r + sr2 - or1) * 0.5 * (
            1 / L - 1 / U + 0.25 * (r - sr2 ** 2 / r) * (1 / (U ** 2) - 1 / (L ** 2)) + 0.5 * np.log(
        L / U) / r)

    I -= np.diag(np.diag(I))
    I = np.sum(I, axis=1)

    # okay, next compute born radii
    offset_radius = radii - offset
    psi = I * offset_radius
    psi_coefficient = 0.8
    psi2_coefficient = 0
    psi3_coefficient = 2.909125

    psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (psi3_coefficient * psi ** 3)

    B = 1 / (1 / offset_radius - np.tanh(psi_term) / radii)

    # finally, compute the three energy terms
    E = 0

    # single particle
    E += np.sum(surface_tension * (radii + 0.14) ** 2 * (radii / B) ** 6)
    E += np.sum(-0.5 * screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / B)

    # particle pair
    f = np.sqrt(r ** 2 + np.outer(B, B) * np.exp(-r ** 2 / (4 * np.outer(B, B))))
    charge_products = np.outer(charges, charges)

    E += np.sum(np.triu(-screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f, k=1))

    return E

if __name__ == '__main__':
    from simtk.openmm import app
    from simtk import openmm as mm
    from simtk import unit

    from openmmtools.testsystems import AlanineDipeptideImplicit

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

    distance_matrix = squareform(pdist(implicit_traj[0]))

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

    autograd_energies = [compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                                      offset, screening, surface_tension,
                                                      solvent_dielectric, solute_dielectric) for distance_matrix in
                         distance_matrices]

    import matplotlib.pyplot as plt
    from bayes_implicit_solvent.utils import remove_top_right_spines
    ax = plt.subplot(1, 1, 1)
    openmm_energies = [v.value_in_unit(unit.kilojoule_per_mole) for v in implicit_only_Us]
    plt.scatter(autograd_energies, openmm_energies)
    plt.xlabel('autograd energies')
    plt.ylabel('openmm energies')
    plt.plot(sorted(autograd_energies), sorted(autograd_energies), color='grey')
    plt.plot(sorted(openmm_energies), sorted(openmm_energies), color='grey')
    remove_top_right_spines(ax)
    plt.savefig('energy-comparison.jpg', dpi=300)

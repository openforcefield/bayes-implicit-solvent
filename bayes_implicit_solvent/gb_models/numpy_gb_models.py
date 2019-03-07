"""For prototyping, since openmm doesn't support gradients w.r.t. per-particle parameters in CustomGBForce
"""

from autograd import numpy as np

from bayes_implicit_solvent.gb_models.modular_gb_models import step, compute_effective_radii_OBC


# first, implement with for loops: later, vectorize
def compute_OBC_energy_reference(distance_matrix, radii, scales, charges,
                                 offset=0.009, screening=138.935484, surface_tension=28.3919551,
                                 solvent_dielectric=78.5, solute_dielectric=1.0,
                                 psi_coefficient=0.8,
                                 psi2_coefficient=0,
                                 psi3_coefficient=2.909125,
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
                                  psi_coefficient=0.8,
                                  psi2_coefficient=0,
                                  psi3_coefficient=2.909125,
                                  ):
    """Replacing for-loops with vectorized operations"""
    N = len(radii)

    # compute effective radii
    effective_radii = compute_effective_radii_OBC(
        distance_matrix=distance_matrix, radii=radii, scales=scales, offset=offset, psi_coefficient=psi_coefficient,
        psi2_coefficient=psi2_coefficient, psi3_coefficient=psi3_coefficient)
    effective_radii_products = np.outer(effective_radii, effective_radii)

    # finally, compute the three energy terms
    E = 0

    # surface area term
    E += np.sum(surface_tension * (radii + 0.14) ** 2 * (radii / effective_radii) ** 6)

    # Delta G_pol, single particle terms: actually, can this be removed?
    E += np.sum(-0.5 * screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / effective_radii)

    # Delta G_pol, particle pair terms
    squared_distances = (distance_matrix + np.eye(N)) ** 2
    f_GB = np.sqrt(
        squared_distances + effective_radii_products * np.exp(-squared_distances / (4 * effective_radii_products)))
    charge_products = np.outer(charges, charges)

    E += np.sum(np.triu(-screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f_GB, k=1))

    return E

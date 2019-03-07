# Figure out how to mix and match components that are successful
# TODO: Search for invariances that are respected in GB but not in explicit-solvent simulations

# Some possible ingredients that could be mixed and matched:
# * different empirical functions f_GB (default f_GB[i,j] = sqrt(r_ij^2 + R_i R_j exp(-r_ij^2 / (4 R_i R_j)) )
# * Different ways to incorporate charges


# Incorporate charge-hydration asymmetry
# Ref: "Introducing charge hydration asymmetry into the Generalized Born model" (Mukhopadhyay, Aguilar, Tolokh, Onufriev, 2014)

# TODO: Figure out what to do about use of non-differentiable `np.sign()` (see discussion in 4.3.1. of Mukhopadhyay's thesis)
# TODO: Figure out how many psi_coefficients to use automatically?

def step(x):
    # return (x > 0)
    return 1.0 * (x >= 0)


from autograd import numpy as np


def compute_effective_radii_OBC(distance_matrix, radii, scales,
                                offset=0.009,
                                psi_coefficient=0.8,
                                psi2_coefficient=0,
                                psi3_coefficient=2.909125,
                                ):
    """Compute effective radii"""
    N = len(radii)
    r = distance_matrix + np.eye(N)  # so I don't have divide-by-zero nonsense
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

    psi_term = (psi_coefficient * psi) + (psi2_coefficient * psi ** 2) + (psi3_coefficient * psi ** 3)

    effective_radii = 1 / (1 / offset_radius - np.tanh(psi_term) / radii)
    return effective_radii


def cha_scaling_correction_components(
        distance_matrix, charges, effective_radii,
        tau=1, RZ_OH=0.058, R_S=0.052, rho_w=0.14):
    """Compute adjustment to

    Parameters
    ----------
    distance_matrix
    charges
    effective_radii
    tau : float
    RZ_OH : float
        This is defined as Q_zz/p, the ratio of quadrupole and dipole moments of a water model...
        Can I redefine this to depend on two parameters instead of collapsing them?
        (TIP4P-Ew: 0.73 angstroms, TIP3P: 0.58 angstroms, TIP5P-E: 0.18 angstroms)
    R_S : float
        shifts the dielectric boundary by 0.52angstroms (p. 45 of Mukhopadhyay's thesis)
    rho_w : float
        Water probe radius, default 1.4angstroms (p. 45 of Mukhopadhyay's thesis)
    Returns
    -------

    """
    N = len(distance_matrix)
    assert (len(charges) == N)
    assert (len(effective_radii) == N)

    charge_influence = np.exp(-tau * (distance_matrix ** 2) / np.outer(effective_radii, effective_radii))
    influences = np.dot(charge_influence, charges)
    # signs = np.sign(influences)
    scales = RZ_OH / (effective_radii - R_S + rho_w)

    # TODO: What if I just remove np.sign? or replace it with np.tanh or something?
    # TODO: Does the sum over j in equation 4 include or exclude i=j?

    return influences, scales


def cha_gb_no_threshold(distance_matrix, radii, scales, taus, charges,
                        offset=0.009, screening=138.935484, surface_tension=28.3919551,
                        solvent_dielectric=78.5, solute_dielectric=1.0,
                        psi_coefficient=0.8,
                        psi2_coefficient=0,
                        psi3_coefficient=2.909125,
                        RZ_OH=0.058, R_S=0.052, rho_w=0.14,
                        ):
    """OBC+CHA with a couple modifications:
    * Allow tau to be a per-particle parameter instead of a global
    * Replace `np.sign(influences) * scales` with `influences * scales`
    """
    N = len(radii)

    # compute effective radii
    effective_radii = compute_effective_radii_OBC(
        distance_matrix=distance_matrix, radii=radii, scales=scales, offset=offset, psi_coefficient=psi_coefficient,
        psi2_coefficient=psi2_coefficient, psi3_coefficient=psi3_coefficient)
    effective_radii_products = np.outer(effective_radii, effective_radii)

    influences, scales = cha_scaling_correction_components(
        distance_matrix, charges, effective_radii,
        tau=taus, RZ_OH=RZ_OH, R_S=R_S, rho_w=rho_w)
    cha_adjusted_radii = effective_radii * (1 + (influences * scales))
    cha_adjusted_radii_products = np.outer(cha_adjusted_radii, cha_adjusted_radii)

    # finally, compute the three energy terms
    E = 0

    # surface area term
    E += np.sum(surface_tension * (radii + 0.14) ** 2 * (radii / effective_radii) ** 6)

    # Delta G_pol, single particle terms: actually, can this be removed?
    E += np.sum(-0.5 * screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charges ** 2 / effective_radii)

    # Delta G_pol, particle pair terms
    squared_distances = (distance_matrix + np.eye(N)) ** 2
    # f_GB = np.sqrt(squared_distances + effective_radii_products * np.exp(-squared_distances / (4 * effective_radii_products)))
    f_GB_CHA = np.sqrt(
        squared_distances + cha_adjusted_radii_products * np.exp(-squared_distances / (4 * effective_radii_products)))

    charge_products = np.outer(charges, charges)

    E += np.sum(
        np.triu(-screening * (1 / solute_dielectric - 1 / solvent_dielectric) * charge_products / f_GB_CHA, k=1))

    return E

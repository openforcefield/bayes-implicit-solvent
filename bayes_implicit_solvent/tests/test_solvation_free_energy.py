from bayes_implicit_solvent.solvation_free_energy import beta, construct_gbsa_force, get_vacuum_samples, create_implicit_sim, get_implicit_u_diffs, predict_solvation_free_energy

import numpy as np

import pytest

from openmmtools.testsystems import AlanineDipeptideVacuum

@pytest.fixture
def alaninedipeptide_testsystem():
    testsystem = AlanineDipeptideVacuum()
    return testsystem

def test_construct_gbsa_force(alaninedipeptide_testsystem):
    """Assert that constructing"""
    system = alaninedipeptide_testsystem.system
    gbsa_force = construct_gbsa_force(system)
    assert("GBSA" in gbsa_force.__class__.__name__)

# TODO: Also test what happens if there already is a GBSA force in system...

def test_get_vacuum_samples(alaninedipeptide_testsystem):
    a = alaninedipeptide_testsystem
    n_samples = 10
    vacuum_sim, vacuum_traj = get_vacuum_samples(a.topology, a.system, a.positions, n_samples=n_samples, thinning=10)
    vacuum_sim.step(1) # check that vacuum_sim has a step method and doesn't crash
    assert (len(vacuum_traj) == n_samples) # check we have the right number of samples
    assert ( len(vacuum_traj[0]) == len(a.positions)) # check the samples have the right shape

from bayes_implicit_solvent.utils import get_gbsa_force
def test_create_implicit_sim(alaninedipeptide_testsystem):
    """Check that we can create a simulation object with a GBSA force in force group 0,
    all other forces in group 1."""
    topology, system = alaninedipeptide_testsystem.topology, alaninedipeptide_testsystem.system
    implicit_sim = create_implicit_sim(topology, system)

    # check that implicit sim contains a GBSA force
    gbsa_force = get_gbsa_force(implicit_sim.system)
    assert ("GBSA" in gbsa_force.__class__.__name__)

    # check that the GBSA force is in group 0
    assert ( gbsa_force.getForceGroup() == 0 )

    # check that all other forces are in group 1
    for force in implicit_sim.system.getForces():
        if 'GBSA' not in force.__class__.__name__:
            assert ( force.getForceGroup() == 1)


def test_get_implicit_u_diffs():
    """check that the function based on force groups computes the same reduced potential energy differences
    as directly computing beta * (U_implicit - U_vacuum)"""

    # TODO: Finish writing this test
    raise (NotImplementedError('must initialize vacuum_sim, implicit_sim, x'))
    vacuum_sim, implicit_sim, x = None, None, None

    u_diff = get_implicit_u_diffs(implicit_sim, x)

    u_diff_ref = np.zeros(len(x))

    for i in range(len(x)):
        vacuum_sim.context.setPositions(x[i])
        implicit_sim.context.setPositions(x[i])

        U_implicit = implicit_sim.context.getState(getEnergy=True).getPotentialEnergy()
        U_vacuum = vacuum_sim.context.getState(getEnergy=True).getPotentialEnergy()
        u_diff_ref[i] = beta * (U_implicit - U_vacuum)

        u_diff_ref[i] = beta * implicit_sim.context.getState(getEnergy=True, groups={0}).getPotentialEnergy()

    np.testing.assert_allclose(u_diff, u_diff_ref)

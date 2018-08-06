"""
Unit and regression test for the bayes_implicit_solvent package.
"""

# Import package, test suite, and other packages as needed
import bayes_implicit_solvent
import pytest
import sys

def test_bayes_implicit_solvent_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "bayes_implicit_solvent" in sys.modules

import numpy as np

from bayes_implicit_solvent.propertycalculator.compute_solvation_free_energy import get_implicit_u_diffs, beta
def test_get_implicit_u_diffs():
    """check that the function based on force groups computes the same reduced potential energy differences
    as directly computing beta * (U_implicit - U_vacuum)"""

    # TODO: Finish writing this test
    raise(NotImplementedError('must initialize vacuum_sim, implicit_sim, x'))
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
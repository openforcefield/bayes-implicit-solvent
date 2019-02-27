"""In this file, we have methods and definitions for whether a typing scheme is legal.


For example, we may require there to be no pair of *lexically identical* types.
We will also enforce that every SMIRKS pattern is valid.

Some checks will make reference to a fixed library of compounds,
for example, checking that a typing scheme doesn't have any unused types or redundant types
"""
from functools import lru_cache

import numpy as np

from bayes_implicit_solvent.freesolv import mol_top_sys_pos_list

all_oe_mols = [entry[0] for entry in mol_top_sys_pos_list]


from openforcefield.typing.engines.smirnoff.forcefield import _validateSMIRKS

# TODO: Replace with minidrugbank or something

def check_valid_smirks(smirks_string):
    return _validateSMIRKS(smirks_string)

def check_all_valid_smirks(typer):
    for smirks in typer.nodes:
        if not check_valid_smirks(smirks):
            return False

    return True


# TODO: Implement check no decorators applied to wildcrd
def check_no_decorators_applied_to_wildcard(typer):
    pass


# TODO: Implement check no duplicates
def check_no_duplicates(typer):
    pass


@lru_cache(maxsize=2 ** 12)
def check_no_empty_types(typer):
    """Apply the typer to every molecule in all_oe_mols, and return -np.inf if the typer contains any
    non-wildcard types that aren't used"""
    assigned_types = typer.apply_to_molecule_list(all_oe_mols)
    flat = np.hstack(assigned_types)
    N = typer.number_of_nodes

    # check that no type is unused
    counts = np.bincount(flat, minlength=N)
    if np.min(counts[1:]) == 0:  # TODO: revisit [1:] slice if we change how wildcard is handled
        # print('empty types found!')
        # print([typer.ordered_nodes[i] for i in range(len(counts)) if counts[i] == 0])
        return -np.inf
    else:
        return 0

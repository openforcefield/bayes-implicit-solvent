# bayes-implicit-solvent
experiments with Bayesian calibration of implicit solvent models

## Contents

### `bayes_implicit_solvent`
* `numpy_gb_models.py` -- Clones the OpenMM GBSA OBC force in numpy (vectorized and non-vectorized), since I wasn't able to compute gradients w.r.t. per-particle parameters in CustomGBForce.
* `posterior_sampling.py` -- Defines a class `Molecule` that predicts solvation free energy as function of GB parameters and compares to an experimental value, for use in posterior sampling.
* `prior_checking.py` -- methods for checking whether a typing scheme is legal
* `samplers.py` -- defines parameter samplers: random-walk Metropolis-Hastings, Langevin (unadjusted and Metropolis-Adjusted), RJMC
* `smarts.py` -- definitions of SMARTS primitives and decorators
* `solvation_free_energy.py` -- functions for computing solvation free energy using GB models
* `typers.py` -- defines the following classes: `DiscreteProposal`, `BondProposal`, `AtomSpecificationProposal`, `BondSpecificationProposal`, `SMIRKSElaborationProposal`, `SMARTSTyper`, `FlatGBTyper`, `GBTypingTree`, which hopefully encapsulate the bookkeeping needed to sample typing schemes using RJMC
* `utils.py` -- un-filed utilities for: interacting with OpenEye, getting or applying GB parameters in OpenMM systems, caching substructure matches

(Currently contains some code that needs to be removed or refactored. `type_samplers.py` defines the following classes: `Proposal`, `RadiusInheritanceProposal`, `AddOrDeletePrimitiveAtEndOfList`, `AddOrDeletePrimitiveAtRandomPositionInList`, `SwapTwoPatterns`, `MultiProposal`, which were used in initial experiments that did not use a tree representation of the typing scheme. `prepare_freesolv.py` uses OpenEye to construct OEMol objects, assign partial charges, etc. starting from a list of SMILES strings.)

#### `bayes_implicit_solvent/continuous-parameter-experiments`
* `elemental_types_mala.py` -- Use Metropolis-adjusted Langevin to sample the radii and scales in the elemtnal-types-only model
* `hydrogen_or_not.py` -- Toy model containing just two "types" -- "hydrogen" vs "not hydrogen" so we can plot the parameter space in 2D for inspection. Attempt to fit GB radii using this restricted typing scheme on subsets of FreeSolv. Also check how the results depend on the number of configuration-samples used in the hydration free energy estimates.
* `smirnoff_types.py` -- Use random-walk Metropolis-Hastings to sample GB radii for models restricted to use the same types defined in the nonbonded force section of smirnoff99frosst.

#### `bayes_implicit_solvent/data`
See its readme: contains freesolv and some numerical results in pickle or numpy archives.

#### `bayes_implicit_solvent/hierarchical_typing`
Out-dated -- initial experiments where types were introduced by truncating the smirnoff nonbonded definitions

#### `bayes_implicit_solvent/tests`
* `test_bayes_implicit_solvent.py`
* `test_rjmc.py` -- unit tests and integration tests that RJMC on typing trees is correct

#### `bayes_implicit_solvent/vacuum_samples`

Scripts to generate configuration samples of FreeSolv set in vacuum, for use in reweighting.

### `devtools`
Copied from MolSSI's `cookiecutter-compchem`. Requirements listed in `devtools/conda-recipe/meta.yaml`.

### `docs`

To-do

### `notebooks`
Exploratory or visualization-focused notebooks.

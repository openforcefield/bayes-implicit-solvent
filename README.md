# bayes-implicit-solvent
experiments with Bayesian calibration of implicit solvent models

## Highlights

### Colab notebook illustrating continuous parameter sampling

Our likelihood function depends on comparing ~600 calculated and experimental hydration free energies, which is computationally expensive and must be done at each sampling iteration.

Gradients of this likelihood are computed efficiently using Jax, and used to compare Langevin Monte Carlo with gradient descent. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//openforcefield/bayes-implicit-solvent/blob/master/notebooks/fast_likelihood_gradients_in_jax_(batches_of_component_gradients).ipynb)

![continuous-parameter-sampling](https://user-images.githubusercontent.com/5759036/68684532-a43c2780-0536-11ea-8144-25a30b106279.png)

### Demonstration of automatic parameterization
A few Markov Chain Monte Carlo algorithms (implemented in [`samplers.py`](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/samplers.py)) are applied to the task of sampling the continuous parameters of implicit solvent models.

![automatic_parameterization](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/continuous_parameter_experiments/automatic_parameterization_figure/automatic_parameterization_draft.png?raw=true)

### Comparisons of Gaussian and Student-t likelihood behavior

One observation from this study has been that the tail behavior of the likelihood function comparing experimental and predicted free energies has a pronounced affect on the behavior of samplers.

![gaussian-vs-student-t](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/RW-MH-RMSE.png?raw=true)

### RJMC experiments
Atom-typing schemes are represented using trees of SMIRKS patterns, implemented in the file [`typers.py`](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/typers.py), along with uniform cross-model proposals that elaborate upon or delete the types within these schemes.

Scripts for numerical experiments with RJMC and various choices of prior, likelihood, within-model sampler, and constraints on the discrete model space are in `bayes_implicit_solvent/rjmc_experiments/`.

Using Langevin Monte Carlo for within-model sampling, and enforcing that elemental types are retained (see the script [`tree_rjmc_w_elements.py`](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/rjmc_experiments/tree_rjmc_w_elements.py)), we obtain the following result.

![rjmc_pilot_figure](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/rjmc_experiments/rjmc_pilot_figure.png?raw=true)

In ongoing work, we are attempting to define better-informed cross-model proposals and use more reasonable prior restraints to improve the chance of converging cross-model sampling. Priors and cross-model proposals that are informed by the number of atoms that fall into each type are being prototyped here [`informed_tree_proposals.py`](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/bayes_implicit_solvent/rjmc_experiments/informed_tree_proposals.py).

### Differentiable atom-typing experiments

As an alternative to assigning parameters using trees of SMIRKS, we also briefly considered assigning parameters using differentiable functions of SMIRKS features.
This would allow uncertainty in the parameter-assignment scheme to be represented using a posterior distribution over continuous variables only, rather than a challenging mixed continuous/discrete space.

Linear functions of SMIRKS fingerprints to radii and scales ([notebook](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/linear-typing-using-atom-features-only-student-t-loss.ipynb))

![linear-typing-student-t](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/linear_parameterizer_student_t_loss/linear_typing_student_t_loss.gif?raw=true)

Multilayer perceptron from SMIRKS fingerprints to radii, scales, and parameters controlling charge-hydration asymmetry ([notebook](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/feedforward-typing-using-smarts-and-neighbor-features-student-t%2Bcha-df%3D7-and-per-particle-psis-big-batch.ipynb)) ![neuralized-cha](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/neuralized_typing_with_cha.gif?raw=true)

(Convolutional typing schemes appeared more difficult to optimize numerically, but may be an interesting direction for future work ([notebook](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/convolutional-typing.ipynb)))

## Detailed contents

### `bayes_implicit_solvent`
* `gb_models/` -- Clones the OpenMM GBSA OBC force using autodiff frameworks such as `jax` and HIPS `autograd`, to allow differentiating w.r.t. per-particle parameters.
* `molecule.py` -- Defines a class `Molecule` that predicts solvation free energy as function of GB parameters and compares to an experimental value, for use in posterior sampling.
* `prior_checking.py` -- methods for checking whether a typing scheme is legal
* `samplers.py` -- defines parameter samplers: random-walk Metropolis-Hastings, Langevin (unadjusted and Metropolis-Adjusted), RJMC
* `smarts.py` -- definitions of SMARTS primitives and decorators
* `solvation_free_energy.py` -- functions for computing solvation free energy using GB models
* `typers.py` -- defines the following classes: `DiscreteProposal`, `BondProposal`, `AtomSpecificationProposal`, `BondSpecificationProposal`, `SMIRKSElaborationProposal`, `SMARTSTyper`, `FlatGBTyper`, `GBTypingTree`, which hopefully encapsulate the bookkeeping needed to sample typing schemes using RJMC
* `utils.py` -- un-filed utilities for: interacting with OpenEye, getting or applying GB parameters in OpenMM systems, caching substructure matches
* `constants.py` -- temperature, unit conventions, etc.

(Currently contains some code that needs to be removed or refactored. `proposals.py` defines the following classes: `Proposal`, `RadiusInheritanceProposal`, `AddOrDeletePrimitiveAtEndOfList`, `AddOrDeletePrimitiveAtRandomPositionInList`, `SwapTwoPatterns`, `MultiProposal`, which were used in initial experiments that did not use a tree representation of the typing scheme. `prepare_freesolv.py` uses OpenEye to construct OEMol objects, assign partial charges, etc. starting from a list of SMILES strings.)

#### `bayes_implicit_solvent/continuous-parameter-experiments/`
* `elemental_types_mala.py` -- Use Metropolis-adjusted Langevin to sample the radii and scales in the elemtnal-types-only model
* `hydrogen_or_not.py` -- Toy model containing just two "types" -- "hydrogen" vs "not hydrogen" so we can plot the parameter space in 2D for inspection. Attempt to fit GB radii using this restricted typing scheme on subsets of FreeSolv. Also check how the results depend on the number of configuration-samples used in the hydration free energy estimates.
* `smirnoff_types.py` -- Use random-walk Metropolis-Hastings to sample GB radii for models restricted to use the same types defined in the nonbonded force section of smirnoff99frosst.

and many more to be documented further

#### `bayes_implicit_solvent/rjmc_experiments/`
* `informed_tree_proposals.py` -- Experiments with constructing guided discrete-model proposals, as well as with defining more effective priors for the discrete models
* `tree_rjmc_start_from_wildcard.py` -- Experiments running RJMC on GB typing trees starting from wildcard type and building up from there.
* `tree_rjmc_w_elements.py` -- Experiments running RJMC on GB typing trees, keeping elemental types as un-delete-able nodes.

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

#### `elaborate_typing_animation/`
* animated GIFs of initial slow typing-tree sampling code (also affected by a bug that was later corrected, where the charges for some molecules were drastically affected incorrectly prepared). The number of types sampled increased much more than expected, and the sampler became slower the more types were present. ([notebook](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/plot%20elaborate%20rjmc%20typing%20trees--%20now%20with%20less%20smirks-overlap!.ipynb))
![image](https://user-images.githubusercontent.com/5759036/68699657-8f20c200-0551-11ea-8528-46289f9d1e40.png)

#### `bugfixed_typing_animation/`
* animated GIF of early tree-RJMC run

#### `extended-sim-projections/`
* projections and of slow-relaxing torsions in some molecules from FreeSolv
![image](https://user-images.githubusercontent.com/5759036/68699790-c0998d80-0551-11ea-8bf8-5bde502a6d29.png)

#### `carboxyl-torsion-plots/`
* diagnostic plots for some slow torsional degrees of freedom involving carboxylic acids, encountered when preparing vacuum samples for use in reweighting-based likelihood estimator

#### `nelder_mead_plots/`
* baseline of using Nelder-Mead simplex minimization rather than gradient-informed optimization or sampling ([notebook](https://github.com/openforcefield/bayes-implicit-solvent/blob/master/notebooks/inspect%20nelder%20mead%20results.ipynb))

![image](https://user-images.githubusercontent.com/5759036/68699968-1706cc00-0552-11ea-86c4-8840f1628f85.png)
![image](https://user-images.githubusercontent.com/5759036/68699996-238b2480-0552-11ea-9a04-00541382c2c0.png)


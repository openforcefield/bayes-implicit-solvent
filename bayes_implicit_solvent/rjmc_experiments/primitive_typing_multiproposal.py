from bayes_implicit_solvent.typers import FlatGBTyper
from bayes_implicit_solvent.proposals import GBModel
import numpy as np
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots


from bayes_implicit_solvent.solvation_free_energy import smiles_list

# initial model: treat all atoms identically
theta_0 = GBModel(typing_scheme=FlatGBTyper(['*']), radii=0.1 * np.ones(1))


# initial subset
np.random.seed(0)
size_of_subset = 100

ind_subset = np.arange(len(smiles_list))
np.random.shuffle(ind_subset)
ind_subset = ind_subset[:size_of_subset]

smiles_subset = [smiles_list[i] for i in ind_subset]

print('"training" on ', smiles_subset)

from pkg_resources import resource_filename
from bayes_implicit_solvent.molecule import Molecule
import mdtraj as md

n_snapshots = 10

mols = []
for (i, smiles) in zip(ind_subset, smiles_subset):
    path_to_samples = resource_filename(
        'bayes_implicit_solvent',
        'vacuum_samples/vacuum_samples_{}.h5'.format(i))
    vacuum_samples = mdtraj_to_list_of_unitted_snapshots(md.load(path_to_samples))
    stride = int(len(vacuum_samples) / n_snapshots)
    mols.append(Molecule(smiles, vacuum_samples=vacuum_samples[::stride]))


# okay let's try adding and removing primitive types!
from bayes_implicit_solvent.smarts import atomic_primitives, atomic_number_dict
from bayes_implicit_solvent.proposals import AddOrDeletePrimitiveAtRandomPositionInList, SwapTwoPatterns, MultiProposal
from bayes_implicit_solvent.samplers import random_walk_mh, sparse_mh

birth_death = AddOrDeletePrimitiveAtRandomPositionInList(list(atomic_primitives.keys()))
swapping = SwapTwoPatterns()

cross_model_proposal = MultiProposal([birth_death, swapping])


def log_prob(gb_model):
    types = [gb_model.typing_scheme.get_gb_types(mol.mol) for mol in mols]
    radii = gb_model.radii
    print('radii: ', radii)
    print('types[0]: ', types[0])

    log_priors = [mol.log_prior(radii) for mol in mols]
    if min(log_priors) > - np.inf:
        atomic_radii = [np.array([radii[t] for t in types[i]]) for i in range(len(mols))]
        # in cases where some of the parameters are unused, they can become negative / fall outside the range of the prior...
        log_prob = sum([mol.log_prob(radii) for (mol, radii) in zip(mols, atomic_radii)])
        return log_prob
    else:
        return - np.inf

theta_traj = [theta_0]
log_ps = [log_prob(theta_0)]
n_cross_model_accept = 0

within_model_trajs = []
within_model_log_ps = []
within_model_acceptance_fractions = []

n_iterations = 1000
for i in range(n_iterations):
    print('iteration', i, 'of', n_iterations, '!')
    proposal_dict = cross_model_proposal.sample_proposal(theta_traj[-1])
    proposal_gb_model = proposal_dict['proposal']

    log_p_proposal = log_prob(proposal_gb_model)
    log_p_f_over_r = proposal_dict['log_p_forward_over_reverse']
    if np.random.rand() < np.exp((log_p_proposal - log_ps[-1]) - log_p_f_over_r):
        n_cross_model_accept += 1
        print('accepted a cross-model-jump!')
        print('\t\t', theta_traj[-1], '-->', proposal_gb_model)
        theta_traj.append(proposal_dict['proposal'])
        log_ps.append(log_p_proposal)


    else:
        theta_traj.append(theta_traj[-1])
        log_ps.append(log_ps[-1])

    # TODO: Wrap this into Multiproposal
    typing_scheme = theta_traj[-1].typing_scheme
    initial_radii = theta_traj[-1].radii
    types = [typing_scheme.get_gb_types(mol.mol) for mol in mols]

    def within_model_log_prob(radii):
        """"""
        log_priors = [mol.log_prior(radii) for mol in mols]
        if min(log_priors) > - np.inf:
            # in cases where some of the parameters are unused, they can become negative / fall outside the range of the prior...

            atomic_radii = [np.array([radii[t] for t in types[i]]) for i in range(len(mols))]
            log_prob = sum([mol.log_prob(radii) for (mol, radii) in zip(mols, atomic_radii)])
            return log_prob
        else:
            return - np.inf

    # sample continuous parameters within each model
    within_model_traj, within_model_log_p, within_model_acceptance_fraction = \
        sparse_mh(initial_radii, within_model_log_prob, n_steps=10, stepsize=0.01, dim_to_perturb=2)

    within_model_trajs.append(within_model_traj)
    within_model_log_ps.append(within_model_log_p)
    within_model_acceptance_fractions.append(within_model_acceptance_fraction)

    theta_traj.append(GBModel(typing_scheme=typing_scheme, radii=within_model_traj[-1]))
    log_ps.append(within_model_log_p[-1])

# save results
experiment_number = 6

np.save('results/experiment_{}_log_ps.npy'.format(experiment_number), log_ps)
radii = [theta.radii for theta in theta_traj]
smarts_lists = [theta.typing_scheme.smarts_list for theta in theta_traj]

from pickle import dump
with open('results/experiment_{}_radii_samples.pkl'.format(experiment_number), 'wb') as f:
    dump(radii, f)

with open('results/experiment_{}_smarts_lists_samples.pkl'.format(experiment_number), 'wb') as f:
    dump(smarts_lists, f)

with open('results/experiment_{}_within_model_trajs.pkl'.format(experiment_number), 'wb') as f:
    dump(within_model_trajs, f)

from bayes_implicit_solvent.typers import GBTyper
from bayes_implicit_solvent.type_samplers import GBModel
import numpy as np
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots


from bayes_implicit_solvent.solvation_free_energy import smiles_list

# initial model: treat all atoms identically
theta_0 = GBModel(typing_scheme=GBTyper(['*']), radii=0.1 * np.ones(1))


# initial subset
np.random.seed(0)

ind_subset = np.random.randint(0, len(smiles_list), 20)
smiles_subset = [smiles_list[i] for i in ind_subset]

from pkg_resources import resource_filename
from bayes_implicit_solvent.posterior_sampling import Molecule
import mdtraj as md

n_snapshots = 50

mols = []
for (i, smiles) in zip(ind_subset, smiles_subset):
    path_to_samples = resource_filename(
        'bayes_implicit_solvent',
        'vacuum_samples/vacuum_samples_{}.h5'.format(i))
    vacuum_samples = mdtraj_to_list_of_unitted_snapshots(md.load(path_to_samples))
    stride = int(len(vacuum_samples) / n_snapshots)
    mols.append(Molecule(smiles, vacuum_samples=vacuum_samples[::stride]))


# okay let's try adding and removing primitive types!
from tqdm import tqdm
from bayes_implicit_solvent.smarts import atomic_primitives, atomic_number_dict
from bayes_implicit_solvent.type_samplers import AddOrDeletePrimitiveAtEndOfList
from bayes_implicit_solvent.samplers import random_walk_mh

# getting error messages that it can't parse most of these smarts strings...
#cross_model_proposal = AddOrDeletePrimitiveAtEndOfList(list(atomic_primitives.keys()))

# no errors so far with this instead
cross_model_proposal = AddOrDeletePrimitiveAtEndOfList(['*'] + list(atomic_number_dict.keys()))


def log_prob(gb_model):
    types = [gb_model.typing_scheme.get_gb_types(mol.mol) for mol in mols]
    radii = gb_model.radii
    print('radii: ', radii)
    print('types[0]: ', types[0])
    atomic_radii = [np.array([radii[t] for t in types[i]]) for i in range(len(mols))]

    return sum([mol.log_prob(radii) for (mol, radii) in zip(mols, atomic_radii)])


theta_traj = [theta_0]
log_ps = [log_prob(theta_0)]
n_cross_model_accept = 0

within_model_trajs = []
within_model_log_ps = []
within_model_acceptance_fractions = []

for i in range(1000):
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

    # TODO: Replace this with caching in typing_scheme object?
    typing_scheme = theta_traj[-1].typing_scheme
    initial_radii = theta_traj[-1].radii
    types = [typing_scheme.get_gb_types(mol.mol) for mol in mols]

    def within_model_log_prob(radii):
        """"""
        atomic_radii = [np.array([radii[t] for t in types[i]]) for i in range(len(mols))]
        return sum([mol.log_prob(radii) for (mol, radii) in zip(mols, atomic_radii)])

    # sample continuous parameters within each model
    within_model_traj, within_model_log_p, within_model_acceptance_fraction = \
        random_walk_mh(initial_radii, within_model_log_prob, n_steps=10, stepsize=0.01)

    within_model_trajs.append(within_model_traj)
    within_model_log_ps.append(within_model_log_p)
    within_model_acceptance_fractions.append(within_model_acceptance_fraction)

    theta_traj.append(GBModel(typing_scheme=typing_scheme, radii=within_model_traj[-1]))
    log_ps.append(within_model_log_p[-1])

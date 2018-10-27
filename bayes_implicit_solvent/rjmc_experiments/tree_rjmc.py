import mdtraj as md
import numpy as np
from pkg_resources import resource_filename
from tqdm import tqdm

from bayes_implicit_solvent.posterior_sampling import Molecule
from bayes_implicit_solvent.smarts import atomic_number_dict
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.typers import GBTypingTree
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'

np.random.seed(0)
initial_tree = GBTypingTree()
for base_type in atomic_number_dict.keys():
    initial_tree.add_child(child_smirks=base_type, parent_smirks='*')

print('initial tree:')
print(initial_tree)

mols = []

smiles_subset = list(smiles_list)
np.random.shuffle(smiles_subset)
smiles_subset = smiles_subset[:int(len(smiles_list) / 10)]
n_configuration_samples = 10

for smiles in smiles_subset:
    mol = Molecule(smiles, vacuum_samples=[])
    path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                               'vacuum_samples/vacuum_samples_{}.h5'.format(
                                                   mol.mol_index_in_smiles_list))
    vacuum_traj = md.load(path_to_vacuum_samples)
    thinning = int(len(vacuum_traj) / n_configuration_samples)
    mol.vacuum_traj = mdtraj_to_list_of_unitted_snapshots(vacuum_traj[::thinning])
    print('thinned vacuum_traj from {} to {}'.format(len(vacuum_traj), len(mol.vacuum_traj)))
    mols.append(mol)


def remove_unit(unitd_quantity):
    """TODO: fix mol.log_prior function so this step isn't necessary"""
    return unitd_quantity / unitd_quantity.unit


def log_prob(tree):
    # TODO: add prior checking, also don't propose so many invalid smarts
    log_prior = 0
    try:
        log_posterior = sum([mol.log_prob(remove_unit(tree.assign_radii(mol.mol))) for mol in mols])
        print('Warning! Encountered un-anticipated exception.')
    except:
        return - np.inf
    # return sum([mol.log_prob(tree.assign_radii(mol.mol)) for mol in mols])
    return log_prior + log_posterior


def tree_rjmc(initial_tree, n_iterations=1000, fraction_cross_model_proposals=0.05):
    trees = [initial_tree]
    log_probs = [log_prob(trees[-1])]
    log_acceptance_probabilities = []

    trange = tqdm(range(n_iterations))
    for _ in trange:
        if np.random.rand() < fraction_cross_model_proposals:
            proposal_dict = trees[-1].sample_create_delete_proposal()
        else:
            proposal_dict = trees[-1].sample_radius_perturbation_proposal()
        log_prob_proposal = log_prob(proposal_dict['proposal'])
        log_p_new_over_old = log_prob_proposal - log_probs[-1]

        log_acceptance_probability = min(0.0, log_p_new_over_old - proposal_dict['log_prob_forward_over_reverse'])
        log_acceptance_probabilities.append(log_acceptance_probability)
        acceptance_probability = min(1.0, np.exp(log_acceptance_probability))
        if np.random.rand() < acceptance_probability:
            trees.append(proposal_dict['proposal'])
            log_probs.append(log_prob_proposal)
        else:
            trees.append(trees[-1])
            log_probs.append(log_probs[-1])

        trange.set_postfix({'avg. accept. prob.': np.mean(np.exp(log_acceptance_probabilities)),
                            'log posterior': log_probs[-1],
                            '# GB types': trees[-1].number_of_nodes,
                            })

    return {'traj': trees,
            'log_probs': np.array(log_probs),
            'log_acceptance_probabilities': np.array(log_acceptance_probabilities)
            }


from pickle import dump

result = tree_rjmc(initial_tree)
with open('tree_rjmc_run_{}.pkl'.format(len(smiles_subset)), 'wb') as f:
    dump(result, f)

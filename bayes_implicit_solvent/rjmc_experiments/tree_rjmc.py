import mdtraj as md
import numpy as np
from pkg_resources import resource_filename
from tqdm import tqdm

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.smarts import atomic_number_dict
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.typers import GBTypingTree
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'

np.random.seed(0)

un_delete_able_types = ['*']
for base_type in atomic_number_dict.keys():
    un_delete_able_types.append(base_type)
initial_tree = GBTypingTree(un_delete_able_types=un_delete_able_types)
for base_type in un_delete_able_types[1:]:
    initial_tree.add_child(child_smirks=base_type, parent_smirks='*')

print('initial tree:')
print(initial_tree)

mols = []

smiles_subset = list(smiles_list)
np.random.shuffle(smiles_subset)
smiles_subset = smiles_subset[:int(len(smiles_list) * 0.75)]
print('looking at {} entries from FreeSolv'.format(len(smiles_subset)))
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

from bayes_implicit_solvent.prior_checking import check_no_empty_types
error_y_trees = []

def log_prob(tree):
    log_prior = check_no_empty_types(tree)

    if log_prior > -np.inf:
        try:
            # TODO: Parallelize. Note that multiprocessing.Pool won't work here because it doesn't play nice with SwigPy objects
            log_prob_components = [mol.log_prob(remove_unit(tree.assign_radii(mol.mol))) for mol in mols]
            log_posterior = sum(log_prob_components)
        except:
            global error_y_trees
            error_y_trees.append(tree)
            print('Warning! Encountered un-anticipated exception!')
            return - np.inf
        # return sum([mol.log_prob(tree.assign_radii(mol.mol)) for mol in mols])
        return log_prior + log_posterior
    else:
        return log_prior

from bayes_implicit_solvent.samplers import tree_rjmc
from pickle import dump

n_iterations = 200000

result = tree_rjmc(initial_tree, log_prob, n_iterations=n_iterations)
with open('elaborate_tree_rjmc_run_n_compounds={}_n_iter={}_gaussian_ll.pkl'.format(len(smiles_subset), n_iterations) , 'wb') as f:
    dump(result, f)

with open('error_y_trees.pkl', 'wb') as f:
    dump(error_y_trees, f)

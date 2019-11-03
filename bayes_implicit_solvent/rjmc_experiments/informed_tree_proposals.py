from jax.config import config

config.update("jax_enable_x64", True)


def get_molecule_list():

    from numpy import load, random
    from simtk import unit

    from bayes_implicit_solvent.molecule import Molecule

    import sys
    def sample_path_to_unitted_snapshots(path_to_npy_samples):
        xyz = load(path_to_npy_samples)
        traj = [snapshot * unit.nanometer for snapshot in xyz]
        return traj

    from glob import glob
    from pkg_resources import resource_filename

    path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                               'vacuum_samples/vacuum_samples_*.npy')
    paths_to_samples = glob(path_to_vacuum_samples)

    # TODO: Consider running on a 90:10 train/test split again...
    paths_to_samples = paths_to_samples  # [:int(0.9*len(paths_to_samples))]

    print('number of molecules being considered: {}'.format(len(paths_to_samples)))

    def extract_cid_key(path):
        i = path.find('mobley_')
        j = path.find('.npy')
        return path[i:j]

    cids = list(map(extract_cid_key, paths_to_samples))
    mols = []

    # TODO: potentially adjust this -- using n_configuration_samples=25 for faster debugging...
    n_configuration_samples = 25

    from bayes_implicit_solvent.freesolv import cid_to_smiles

    from bayes_implicit_solvent.constants import beta
    def unreduce(value):
        """Input value is in units of kB T, turn it into units of kilocalorie_per_mole"""
        return value / (beta * unit.kilocalorie_per_mole)

    for path in paths_to_samples:
        cid = extract_cid_key(path)
        smiles = cid_to_smiles[cid]
        vacuum_samples = sample_path_to_unitted_snapshots(path)
        thinning = int(len(vacuum_samples) / n_configuration_samples)
        mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning])

        if (unreduce(mol.experimental_value) > -15) and (unreduce(mol.experimental_value) < 5):
            mols.append(mol)
        else:
            print(
                'discarding {} ({}) because its free energy was outside of the range [-15, +5] kcal/mol'.format(smiles,
                                                                                                                cid))

    oemols = [mol.mol for mol in mols]
    return oemols



from bayes_implicit_solvent.gb_models.obc2_parameters import mbondi_model

tree = mbondi_model
tree.remove_node('[#14]')
tree.add_child('[#1]~[#8]', '[#1]')
tree.add_child('[#1]~[#6]', '[#1]')
oemols = get_molecule_list()
types = tree.apply_to_molecule_list(oemols)

import numpy as onp

def get_probabilities_of_elaborating(tree):
    """elaborate on a type proportional to the number of atoms it hits,
    mask by whether elaborate-able
    """
    types = tree.apply_to_molecule_list(oemols)
    counts = onp.bincount(onp.hstack(types))
    decorate_able_mask = onp.array(list(map(tree.is_decorate_able, tree.ordered_nodes)))
    counts_ = counts * decorate_able_mask
    del(counts)
    return counts_ / onp.sum(counts_)


def get_probabilities_of_removing(tree):
    """delete a type inversely proportional to the number of atoms it hits,
    mask by whether delete-able (for example, only leaf nodes are ever delete-able)"""
    types = tree.apply_to_molecule_list(oemols)
    counts = onp.bincount(onp.hstack(types))
    delete_able_mask = onp.array(list(map(tree.is_delete_able, tree.ordered_nodes)))
    counts_ = counts * delete_able_mask
    del(counts)
    return counts_ / onp.sum(counts_)


for _ in range(100):
    tree_proposal = tree.sample_create_delete_proposal()
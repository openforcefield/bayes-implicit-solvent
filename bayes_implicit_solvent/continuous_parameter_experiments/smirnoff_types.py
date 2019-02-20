from bayes_implicit_solvent.typers import GBTypingTree, AtomSpecificationProposal
specifiers = ['X1', 'X2', 'X3', 'X4']
atom_specification_proposal = AtomSpecificationProposal(atomic_specifiers=specifiers)
typer = GBTypingTree(atom_specification_proposal)

# hydrogen types
typer.add_child('[#1]', '*')
typer.add_child('[#1]-[#6X4]', '[#1]')
#typer.add_child('[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X4]')
#typer.add_child('[#1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]',
#                '[#1]-[#6X4]-[#7,#8,#9,#16,#17,#35]')
#typer.add_child('[#1]-[#6X4]~[*+1,*+2]', '[#1]-[#6X4]')
#typer.add_child('[#1]-[#6X3]', '*')
#typer.add_child('[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]')
#typer.add_child('[#1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]', '[#1]-[#6X3]~[#7,#8,#9,#16,#17,#35]')
typer.add_child('[#1]-[#6X2]', '[#1]')
typer.add_child('[#1]-[#7]', '[#1]')
typer.add_child('[#1]-[#8]', '[#1]')
typer.add_child('[#1]-[#16]', '[#1]')

# carbon types
typer.add_child('[#6]', '*')
typer.add_child('[#6X2]', '[#6]')
typer.add_child('[#6X4]', '[#6]')

# nitrogen type
typer.add_child('[#7]', '*')

# oxygen types
typer.add_child('[#8]', '*')
typer.add_child('[#8X2H0+0]', '[#8]')
typer.add_child('[#8X2H1+0]', '[#8]')

# fluorine types
typer.add_child('[#9]', '*')

# phosphorus type
typer.add_child('[#15]', '*')

# sulfur type
typer.add_child('[#16]', '*')

# chlorine type
typer.add_child('[#17]', '*')

# bromine type
typer.add_child('[#35]', '*')

# iodine type
typer.add_child('[#53]', '*')

print(typer)

from bayes_implicit_solvent.prior_checking import check_no_empty_types

check_no_empty_types(typer)

import os.path

import mdtraj as md
import numpy as np
from pkg_resources import resource_filename

from bayes_implicit_solvent.molecule import Molecule
from bayes_implicit_solvent.samplers import sparse_mh
from bayes_implicit_solvent.solvation_free_energy import smiles_list
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

data_path = '../data/'

np.random.seed(0)

inds = np.arange(len(smiles_list))
#np.random.shuffle(inds)
inds = inds[:int(len(smiles_list) / 2)]

smiles_subset = [smiles_list[i] for i in inds]

n_configuration_samples = 10

mols = []

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

type_assignments = typer.apply_to_molecule_list([mol.mol for mol in mols])
radii0 = np.ones(typer.number_of_nodes) * 0.12
scales0 = np.ones(typer.number_of_nodes) * 0.85

def pack(radii, scales):
    n = len(radii)
    theta = np.zeros(2 * n)
    theta[:n] = radii
    theta[n:2 * n] = scales
    return theta

theta0 = pack(radii0, scales0)

def unpack(theta):
    n = int((len(theta)) / 2)
    radii, scales = theta[:n], theta[n:2 * n]
    return radii, scales

def log_prob(theta):
    """Fixed typing scheme, only radii"""
    radii, scales, = unpack(theta)
    # TODO: Update example to allow variable scale-factors also
    return sum([mols[i].log_prob(radii[type_assignments[i]], scales[type_assignments[i]]) for i in range(len(mols))])


traj, log_probs, acceptance_fraction = sparse_mh(theta0, log_prob, n_steps=10000, dim_to_perturb=3, stepsize=0.01)

np.savez(os.path.join(data_path,
                     'smirnoff_type_radii_samples_half_of_freesolv_n_config={}.npy'.format(
                         n_configuration_samples)), traj=traj, log_probs=log_probs)

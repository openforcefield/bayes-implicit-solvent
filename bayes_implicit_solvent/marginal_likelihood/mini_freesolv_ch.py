"""103 compounds containing only carbon and hydrogen
ethane
prop-1-yne
benzene
heptane
9,10-dihydroanthracene
2-methylprop-1-ene
2,3-dimethylnaphthalene
octane
cyclopentane
hept-1-ene
1-isopropyl-4-methyl-benzene
2,3-dimethylbutane
2,6-dimethylnaphthalene
hex-1-yne
butadiene
isobutylbenzene
cyclohexene
oct-1-ene
butylbenzene
isopropenylbenzene
1,4-dimethylcyclohexane
acenaphthene
hexylbenzene
2,2-dimethylpentane
cumene
sec-butylbenzene
2-methylbut-2-ene
1-ethyl-4-methyl-benzene
1-methylcyclohexene
toluene
p-xylene
but-1-yne
propylbenzene
2,4-dimethylpentane
hex-1-ene
anthracene
propane
hept-1-yne
hexane
n-pentane
decane
3,3-dimethylpentane
non-1-ene
styrene
1-ethyl-2-methylbenzene
pyrene
1-methylnaphthalene
2,2,4-trimethylpentane
methylcyclopentane
mesitylene
9H-fluorene
ethylene
pentylcyclopentane
2,2-dimethylbutane
nonane
2-methylhexane
pent-1-ene
1,2,3-trimethylbenzene
m-xylene
methane
1,4-dimethylnaphthalene
but-1-ene
1,3-dimethylnaphthalene
phenanthrene
oct-1-yne
tert-butylbenzene
neopentane
3-methylbut-1-ene
isoprene
cyclopentene
isopentane
3-methylhexane
hept-2-ene
2,3-dimethylpentane
2-methylpent-1-ene
1,2,4-trimethylbenzene
2,2,5-trimethylhexane
2,3,4-trimethylpentane
1,2-dimethylcyclohexane
2,3-dimethylbuta-1,3-diene
3-methylpentane
propylcyclopentane
methylcyclohexane
biphenyl
isobutane
cyclopropane
indane
hexa-1,5-diene
pent-1-yne
n-butane
isohexane
3-methylheptane
pentylbenzene
ethylbenzene
1-ethylnaphthalene
cyclohepta-1,3,5-triene
penta-1,4-diene
pent-2-ene
naphthalene
1,1-diphenylethene
o-xylene
cyclohexane
prop-1-ene"""

import numpy as np
from simtk import unit

from bayes_implicit_solvent.molecule import Molecule


def sample_path_to_unitted_snapshots(path_to_npy_samples):
    xyz = np.load(path_to_npy_samples)
    traj = [snapshot * unit.nanometer for snapshot in xyz]
    return traj


from glob import glob
from pkg_resources import resource_filename

ll = 'student-t'
n_conf = 5

path_to_vacuum_samples = resource_filename('bayes_implicit_solvent',
                                           'vacuum_samples/vacuum_samples_*.npy')
paths_to_samples = glob(path_to_vacuum_samples)


def extract_cid_key(path):
    i = path.find('mobley_')
    j = path.find('.npy')
    return path[i:j]


cids = list(map(extract_cid_key, paths_to_samples))

mols = []

n_configuration_samples = n_conf

from bayes_implicit_solvent.freesolv import cid_to_smiles

for path in paths_to_samples:
    cid = extract_cid_key(path)
    smiles = cid_to_smiles[cid]
    vacuum_samples = sample_path_to_unitted_snapshots(path)
    thinning = int(len(vacuum_samples) / n_configuration_samples)
    mol = Molecule(smiles, vacuum_samples=vacuum_samples[::thinning], ll=ll)

    if set([a.element.symbol for a in mol.top.atoms()]) == {'C', 'H'}:
        print(mol.mol_name)
        mols.append(mol)


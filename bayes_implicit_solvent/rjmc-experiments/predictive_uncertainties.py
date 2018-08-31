from bayes_implicit_solvent.utils import remove_top_right_spines
from bayes_implicit_solvent.typers import GBTyper
from bayes_implicit_solvent.type_samplers import GBModel

from pickle import load
import matplotlib.pyplot as plt
import numpy as np

experiment_number = 3

with open('experiment_{}_radii_samples.pkl'.format(experiment_number), 'rb') as f:
    radii_samples = load(f)

with open('experiment_{}_smarts_lists_samples.pkl'.format(experiment_number), 'rb') as f:
    smarts_lists_samples = load(f)

gb_models = [GBModel(GBTyper(smarts_lists_samples[i]), radii_samples[i]) for i in range(len(radii_samples))]


from bayes_implicit_solvent.solvation_free_energy import smiles_list, db
ind_test = np.random.randint(len(smiles_list))
smiles_test = smiles_list[ind_test]

for entry in db:
    if entry[1] == smiles_test:
        name_test = entry[2]

print('testing on ', name_test, " ({})".format(smiles_test))

from pkg_resources import resource_filename
from bayes_implicit_solvent.posterior_sampling import Molecule
import mdtraj as md

n_snapshots = 10
from bayes_implicit_solvent.utils import mdtraj_to_list_of_unitted_snapshots

path_to_samples = resource_filename(
    'bayes_implicit_solvent',
    'vacuum_samples/vacuum_samples_{}.h5'.format(ind_test))
vacuum_samples = mdtraj_to_list_of_unitted_snapshots(md.load(path_to_samples))
stride = int(len(vacuum_samples) / n_snapshots)
predict_mol = Molecule(smiles_test, vacuum_samples=vacuum_samples[::stride])

means, uncs = [], []
from tqdm import tqdm
for gb_model in tqdm(gb_models):
    types = gb_model.typing_scheme.get_gb_types(predict_mol.mol)
    radii = np.array([gb_model.radii[t] for t in types])
    mean, unc = predict_mol.predict_solvation_free_energy(radii)
    means.append(mean)
    uncs.append(unc)


burn_in = 100

plt.errorbar(np.arange(len(means))[burn_in:], means[burn_in:], uncs[burn_in:])
plt.hlines(predict_mol.experimental_value, burn_in, len(means), linestyles='--')
plt.fill_between(np.arange(len(means))[burn_in:],
                 [predict_mol.experimental_value - 1.92 * predict_mol.experimental_uncertainty] * (len(means) - burn_in),
                 [predict_mol.experimental_value + 1.92 * predict_mol.experimental_uncertainty] * (len(means) - burn_in),
                 alpha=0.1)
plt.xlabel('RJMC iteration')
plt.ylabel('hydration free energy (in kT)')
plt.title(name_test)
plt.savefig('predicted_delta_G.png', dpi=300)
plt.close()

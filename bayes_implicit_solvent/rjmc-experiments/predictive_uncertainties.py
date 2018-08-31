from bayes_implicit_solvent.utils import remove_top_right_spines
from bayes_implicit_solvent.typers import GBTyper
from bayes_implicit_solvent.type_samplers import GBModel

from pickle import load
import matplotlib.pyplot as plt
import numpy as np

experiment_number = 4

with open('results/experiment_{}_radii_samples.pkl'.format(experiment_number), 'rb') as f:
    radii_samples = load(f)

with open('results/experiment_{}_smarts_lists_samples.pkl'.format(experiment_number), 'rb') as f:
    smarts_lists_samples = load(f)

gb_models = [GBModel(GBTyper(smarts_lists_samples[i]), radii_samples[i]) for i in range(len(radii_samples))]

# TODO: Do this for the whole FreeSolv set, look at "generalization" to molecules not in training set
# (experiment 3 is on a small training set of 20 molecules)
# (experiment 4 is on a larger training set of 300 molecules...)
from bayes_implicit_solvent.solvation_free_energy import smiles_list, db
np.random.seed(100)


for _ in range(10):
    ind_test = np.random.randint(len(smiles_list))
    smiles_test = smiles_list[ind_test]

    for entry in db:
        if entry[1] == smiles_test:
            name_test = entry[2]

    print('testing on ', name_test, " ({})".format(smiles_test))

    from pkg_resources import resource_filename
    from bayes_implicit_solvent.posterior_sampling import Molecule
    import mdtraj as md

    n_snapshots = 100
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

    ax = plt.subplot(1, 2, 1)
    plt.errorbar(np.arange(len(means))[burn_in:], means[burn_in:], uncs[burn_in:])
    plt.hlines(predict_mol.experimental_value, burn_in, len(means), linestyles='--')
    plt.fill_between(np.arange(len(means))[burn_in:],
                     [predict_mol.experimental_value - 1.92 * predict_mol.experimental_uncertainty] * (len(means) - burn_in),
                     [predict_mol.experimental_value + 1.92 * predict_mol.experimental_uncertainty] * (len(means) - burn_in),
                     alpha=0.1)
    plt.xlabel('RJMC iteration')
    plt.ylabel('hydration free energy (in kT)')
    plt.title(name_test + '\ntrace of predictions during sampling')
    remove_top_right_spines(ax)

    ax = plt.subplot(1, 2, 2)
    hist = plt.hist(means[burn_in:], normed=True, bins=30, alpha=0.5)
    heights = hist[0]
    plt.vlines(predict_mol.experimental_value, min(heights), max(heights), linestyles='--')
    plt.fill_betweenx([min(heights), max(heights)],
                     [predict_mol.experimental_value - 1.92 * predict_mol.experimental_uncertainty] * 2,
                     [predict_mol.experimental_value + 1.92 * predict_mol.experimental_uncertainty] * 2,
                     alpha=0.1)

    plt.xlabel('hydration free energy (in kT)')
    plt.ylabel('probability density')
    plt.title(name_test + '\nmarginal predictive distribution')
    plt.yticks([])

    remove_top_right_spines(ax)


    plt.tight_layout()
    plt.savefig('figures/experiment_{}_predicted_delta_G_trace_and_distribution_{}.png'.format(experiment_number, name_test), dpi=300, bbox_inches='tight')
    plt.close()

    # TODO: Replace histogram with nice smooth mixture-of-Gaussians curve...

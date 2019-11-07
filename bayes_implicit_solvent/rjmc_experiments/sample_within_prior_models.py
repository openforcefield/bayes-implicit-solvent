from bayes_implicit_solvent.rjmc_experiments.tree_rjmc_w_elements import *

path_to_model_samples = 'prior_samples_alpha=3.0.pkl'


from pickle import load
with open(path_to_model_samples, 'rb') as f:
    prior_samples = load(f)
print('loaded {} prior samples from {}'.format(len(prior_samples), path_to_model_samples))


if __name__ == "__main__":
    try:
        job_id = int(sys.argv[1])
    except:
        print("Didn't parse input, selecting job parameters at random")
        job_id = random.randint(10000)
    onp.random.seed(job_id)

    tree_ind = onp.random.randint(len(prior_samples))
    tree = prior_samples[tree_ind]
    print('selected the following tree (i={}) from prior sample trajectory'.format(tree_ind))
    print(tree)


    n_chunks = 1000
    n_steps_per_chunk = 50

    trajs = []

    from tqdm import tqdm

    trange = tqdm(range(n_chunks))

    within_model_trajs = []
    prediction_traj = []

    from bayes_implicit_solvent.samplers import langevin

    train_smiles = [mol.smiles for mol in mols]

    def save():
        name = 'within_model_sampling'.format(
            n_iterations,
            ll,
            job_id
        )
        onp.savez(name + '.npz',
                 ll=ll,
                 job_id=job_id,
                 train_smiles=onp.array(train_smiles),
                 tree_ind=tree_ind,
                 within_model_trajs=within_model_trajs,
                 expt_means=expt_means,
                 expt_uncs=expt_uncertainties,
                 predictions=prediction_traj,
                 )
    types = tree.apply_to_molecule_list(oemols)
    theta = get_theta(tree)
    N = int(len(theta) / 2)
    stepsize = 0.001


    def within_model_log_prob(theta):
        return log_posterior(theta, types)


    def within_model_grad_log_prob(theta):
        return grad_log_posterior(theta, types)


    def run_langevin(theta0, stepsize=stepsize):
        v0 = onp.random.randn(*theta0.shape)
        within_model_traj = langevin(theta0, v0, within_model_log_prob, within_model_grad_log_prob,
                                     n_steps=n_steps_per_chunk,
                                     stepsize=stepsize,
                                     collision_rate=0.001 / stepsize)
        current_log_prob = within_model_log_prob(within_model_traj[-1])
        return within_model_traj, current_log_prob


    for chunk in trange:
        within_model_traj, current_log_prob = run_langevin(theta, stepsize)
        while not np.isfinite(current_log_prob):
            print("that didn't go well! trying again with smaller stepsize...")
            print("\told stepsize: ", stepsize)
            stepsize *= 0.5
            print("\tnew stepsize ", stepsize)
            within_model_traj, current_log_prob = run_langevin(theta0, stepsize)

        theta = within_model_traj[-1]

        predictions = get_predictions(within_model_traj[-1], types)
        prediction_traj.append(predictions)
        train_rmse = get_rmse_in_kcal_per_mol(predictions)

        trange.set_postfix(
            current_log_prob=current_log_prob,
            current_train_rmse=train_rmse,
        )

        for t in within_model_traj:
            within_model_trajs.append(t)

        if (chunk + 1) % 20 == 0:
            save()

    save()

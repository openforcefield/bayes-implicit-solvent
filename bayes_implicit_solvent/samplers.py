import numpy as np
from tqdm import tqdm


def random_walk_mh(x0, log_prob_fun, n_steps=1000, stepsize=0.1):
    """Random-walk Metropolis-Hastings with Gaussian proposals.

    Parameters
    ----------
    x0 : array of floats (dim := len(x0))
        initial state of sampler
    log_prob_fun : callable, accepts an array and returns a float
        unnormalized log probability density function
    n_steps : integer
        number of MCMC steps
    stepsize : float
        standard deviation of random walk proposal distribution

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats
        trajectory of samples generated by MCMC
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    acceptance_fraction : float in [0,1]
        fraction of accepted proposals
    """
    dim = len(x0)

    traj = [x0]
    log_probs = [log_prob_fun(x0)]

    acceptances = 0
    r = range(n_steps)
    trange = tqdm(r)
    for n in trange:

        x_proposal = traj[-1] + stepsize * np.random.randn(dim)
        log_prob_proposal = log_prob_fun(x_proposal)

        if np.random.rand() < np.exp(log_prob_proposal - log_probs[-1]):
            traj.append(x_proposal)
            log_probs.append(log_prob_proposal)
            acceptances += 1
        else:
            traj.append(traj[-1])
            log_probs.append(log_probs[-1])

        trange.set_postfix({'log_prob': log_probs[-1], 'accept_fraction': float(acceptances) / (1 + n)})
    del (trange)

    return np.array(traj), np.array(log_probs), float(acceptances) / n_steps


def sparse_mh(x0, log_prob_fun, n_steps=1000, stepsize=0.1, dim_to_perturb=2):
    """Random-walk Metropolis-Hastings with masked Gaussian proposals.
    Only perturbs dim_to_perturb randomly selected dimensions per step.

    Parameters
    ----------
    x0 : array of floats (dim := len(x0))
        initial state of sampler
    log_prob_fun : callable, accepts an array and returns a float
        unnormalized log probability density function
    n_steps : integer
        number of MCMC steps
    stepsize : float
        standard deviation of random walk proposal distribution
    dim_to_perturb : int
        maximum number of components to perturb during proposal

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats
        trajectory of samples generated by MCMC
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    acceptance_fraction : float in [0,1]
        fraction of accepted proposals
    """
    dim = len(x0)

    traj = [x0]
    log_probs = [log_prob_fun(x0)]

    acceptances = 0
    r = range(n_steps)
    trange = tqdm(r)

    for n in trange:
        inds = np.arange(dim)
        np.random.shuffle(inds)
        inds_to_perturb = inds[:dim_to_perturb]

        x_proposal = np.array(traj[-1])
        x_proposal[inds_to_perturb] += stepsize * np.random.randn(len(inds_to_perturb))
        log_prob_proposal = log_prob_fun(x_proposal)

        if np.random.rand() < np.exp(log_prob_proposal - log_probs[-1]):
            traj.append(x_proposal)
            log_probs.append(log_prob_proposal)
            acceptances += 1
        else:
            traj.append(traj[-1])
            log_probs.append(log_probs[-1])

        trange.set_postfix({'log_prob': log_probs[-1], 'accept_fraction': float(acceptances) / (1 + n)})
    del (trange)

    return np.array(traj), np.array(log_probs), float(acceptances) / n_steps


def langevin(x0, v0, log_prob_fun, grad_log_prob_fun, n_steps=100, stepsize=0.01, collision_rate=1e-5):
    """

    Parameters
    ----------
    x0 : array of floats
        initial configuration
    v0 : array of floats
        initial velocities
    log_prob_fun : callable, accepts an array and returns a float
        unnormalized log probability density function
    grad_log_prob_fun : callable, accepts an array and returns an array
        gradient of log_prob_fun
    n_steps : integer
        number of Langevin steps
    stepsize : float > 0
        finite timestep parameter
    collision_rate : float > 0
        controls the rate of interaction with the heat bath

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats
        trajectory of samples generated by Langevin dynamics
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    """
    x = np.array(x0)
    v = np.array(v0)
    traj = [np.array(x)]

    log_probs = [log_prob_fun(x)]

    force = lambda x: -grad_log_prob_fun(x)

    a = np.exp(- collision_rate * stepsize)
    b = np.sqrt(1 - np.exp(-2 * collision_rate * stepsize))

    F = force(x)
    # print(F)

    trange = tqdm(range(n_steps))
    for _ in trange:
        # v
        v += (stepsize * 0.5) * F
        # r
        x += (stepsize * 0.5) * v
        # o
        v = (a * v) + (b * np.random.randn(*x.shape))
        # r
        x += (stepsize * 0.5) * v

        F = force(x)
        # v
        v += (stepsize * 0.5) * F

        log_prob = log_prob_fun(x)
        trange.set_postfix({'log_prob': log_prob})
        # print(energy)

        if (np.sum(np.isfinite(x)) != len(x)) or (not np.isfinite(log_prob)):
            print("Numerical instability encountered!")
            return np.array(traj)

        traj.append(np.array(x))
        log_probs.append(log_prob)

    return np.array(traj), np.array(log_probs)


def MALA(x0, log_prob_fun, grad_log_prob_fun, n_steps=100, stepsize=0.01,
         adapt_stepsize=False, adaptation_factor=0.5, adaptation_interval=50):
    """Metropolis-Adjusted Langevin Algorithm.

    Parameters
    ----------
    x0 : array of floats
        initial configuration
    log_prob_fun : callable, accepts an array and returns a float
        unnormalized log probability density function
    grad_log_prob_fun : callable, accepts an array and returns an array
        gradient of log_prob_fun
    n_steps : integer
        number of Langevin steps
    stepsize : float > 0
        finite timestep parameter
    adapt_stepsize : boolean
        apply a heuristic to increase or decrease the stepsize on the fly
    adaptation_factor : float
        if adapt_stepsize == True, then this parameter sets how big the adjustments are...
        if the stepsize is too small, adapt by t *= (1 + adaptation_factor)
        if too small, adapt by stepsize *= (1 - adaptation_factor)
    adaptation_interval : int
        if adapt_stepsize == True, then this parameter sets how frequent the adjustments are...
        how many sampling steps to attempt before checking whether to adapt stepsize

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats
        trajectory of samples generated by Metropolis-Adjusted Langevin dynamics
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    grads : [n_steps + 1 x dim] array of floats
        trajectory of gradients of the log probability density at the samples generated by Langevin dynamics
    acceptance_probs : [n_steps] array of floats
        acceptance probabilities for the proposals
    stepsizes : [n_steps + 1] array of floats
        stepsizes used at each iteration

    References
    ----------
    [1] G. O. Roberts and J. S. Rosenthal (1998).
        "Optimal scaling of discrete approximations to Langevin diffusions".
        Journal of the Royal Statistical Society, Series B. 60 (1): 255-268.
        doi:10.1111/1467-9868.00123.
    """

    traj = [np.array(x0)]
    log_probs = [log_prob_fun(traj[-1])]
    grads = [grad_log_prob_fun(traj[-1])]
    acceptance_probs = []
    stepsizes = [stepsize]


    def proposal_log_probability(proposal, initial, grad_initial, stepsize):
        return (-1 / (4 * stepsize)) * np.sum((proposal - initial - stepsize * grad_initial) ** 2)

    trange = tqdm(range(n_steps))
    for t in trange:
        sigma = np.sqrt(2 * stepsize)
        proposal = traj[-1] + (stepsize * grads[-1]) + (sigma * np.random.randn(*traj[-1].shape))
        log_prob_proposal = log_prob_fun(proposal)
        grad_proposal = grad_log_prob_fun(proposal)

        log_prob_ratio = log_prob_proposal - log_probs[-1]
        if not np.isfinite(log_prob_proposal):
            acceptance_prob = 0
        else:
            log_forward_proposal_probability = proposal_log_probability(proposal, traj[-1], grads[-1], stepsize)
            log_reverse_proposal_probability = proposal_log_probability(traj[-1], proposal, grad_proposal, stepsize)

            log_acceptance_probability = min(0,
                                             log_prob_ratio + log_reverse_proposal_probability - log_forward_proposal_probability)
            acceptance_prob = np.exp(log_acceptance_probability)
        acceptance_probs.append(acceptance_prob)

        if np.random.rand() < acceptance_prob:
            traj.append(proposal)
            log_probs.append(log_prob_proposal)
            grads.append(grad_proposal)
        else:
            traj.append(traj[-1])
            log_probs.append(log_probs[-1])
            grads.append(grads[-1])

        trange.set_postfix({
            'log_prob': log_probs[-1],
            'average_accept_prob': np.mean(acceptance_probs),
        })

        if (t > 1) and (t % adaptation_interval == 0):
            recent_acceptance_probs = acceptance_probs[-adaptation_interval:]
            if np.mean(recent_acceptance_probs) < 0.1 or np.mean(recent_acceptance_probs) > 0.9:
                print(
                    'Acceptance rate recently ({}) is not close to the optimal acceptance rate (0.574). Consider adjusting the step-size.'.format(
                        np.mean(acceptance_probs)))
            if adapt_stepsize:
                print('Attempting to adapt the step-size automatically. This may be a bad idea.')
                if np.mean(acceptance_probs[-adaptation_interval:]) < 0.1:
                    # acceptance probability too low, step size probably too big, try shrinking it
                    new_stepsize = stepsize * (1 - adaptation_factor)
                else:
                    # acceptance probability too high, step size probably too small, try increasing it
                    new_stepsize = stepsize * (1 + adaptation_factor)
                print('updating stepsize from {} to {}'.format(stepsize, new_stepsize))
                stepsize = new_stepsize

        stepsizes.append(stepsize)

    return np.array(traj), np.array(log_probs), np.array(grads), np.array(acceptance_probs), np.array(stepsizes)

# TODO: Automatically adjust MALA stepsize


def tree_rjmc(initial_tree, log_prob_func, n_iterations=1000, fraction_cross_model_proposals=0.25):
    trees = [initial_tree]
    log_probs = [log_prob_func(trees[-1])]
    log_acceptance_probabilities = []

    trange = tqdm(range(n_iterations))
    for _ in trange:
        if np.random.rand() < fraction_cross_model_proposals:
            proposal_dict = trees[-1].sample_create_delete_proposal()

        else:
            proposal_dict = trees[-1].sample_radius_perturbation_proposal()
        log_prob_proposal = log_prob_func(proposal_dict['proposal'])
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

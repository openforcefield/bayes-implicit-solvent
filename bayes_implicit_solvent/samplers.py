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

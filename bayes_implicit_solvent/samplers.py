import numpy as np
from tqdm import tqdm

def random_walk_mh(x0, log_prob_fun, n_steps=1000, stepsize=0.1):
    dim = len(x0)

    traj = [x0]
    log_probs = [log_prob_fun(x0)]

    acceptances = 0
    r = range(n_steps)
    trange = tqdm(r)
    # for n in r:
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

from bayes_implicit_solvent.utils import remove_top_right_spines

from pickle import load

experiment_number = 1

with open('experiment_{}_radii_samples.pkl'.format(experiment_number), 'rb') as f:
    radii_samples = load(f)
import matplotlib.pyplot as plt
import numpy as np

log_ps = np.load('experiment_{}_log_ps.npy'.format(experiment_number))

n_types_trace = [len(r) for r in radii_samples]

fig = plt.figure(figsize=(12,4))


traces = []
for i in range(1, 5):
    trace = []
    for r in radii_samples:
        if len(r) == i:
            trace.append(r)
    traces.append(np.array(trace))

for i in range(1, 5):
    plt.subplot(4,1,i)
    for component in range(i):
        plt.hist(traces[i][:,component], bins=50, normed=True, histtype='stepfilled', alpha=0.3)
plt.savefig('within_model_histograms.png')
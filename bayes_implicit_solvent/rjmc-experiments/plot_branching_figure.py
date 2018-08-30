from bayes_implicit_solvent.utils import remove_top_right_spines

from pickle import load

experiment_number = 1

with open('experiment_{}_radii_samples.pkl'.format(experiment_number), 'rb') as f:
    radii_samples = load(f)
import matplotlib.pyplot as plt
import numpy as np

log_ps = np.load('experiment_{}_log_ps.npy'.format(experiment_number))

n_types_trace = [len(r) for r in radii_samples]

max_n_dimensions = max(n_types_trace)
print(max_n_dimensions)

fig = plt.figure(figsize=(12,4))


traces = []
for i in range(max_n_dimensions):
    trace = []
    for r in radii_samples:
        if len(r) > i:
            trace.append(r[i])
        else:
            trace.append(np.nan)
    traces.append(trace)

ax = plt.subplot(3,1,1)
# plot branching
for trace in traces:
    plt.plot(trace)
#plt.xlabel('iteration')
plt.ylabel('radius')
remove_top_right_spines(ax)

# plot # types trace
ax = plt.subplot(3,1,2)
plt.plot(n_types_trace)
#plt.xlabel('iteration')
plt.ylabel('# GB types')
plt.yticks([1,10,20])
remove_top_right_spines(ax)

# plot log-probability trace
ax = plt.subplot(3,1,3)
plt.plot(log_ps)
plt.xlabel('iteration')
plt.ylim(-1500, max(log_ps) + 20)
plt.ylabel('log posterior')
remove_top_right_spines(ax)

plt.tight_layout()

plt.savefig('experiment_{}_branching.png'.format(experiment_number),
            bbox_inches='tight', dpi=300)
plt.close()
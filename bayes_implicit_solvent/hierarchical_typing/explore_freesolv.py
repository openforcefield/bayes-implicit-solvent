# random functions that examine the Freesolv database -- outside the main flow of the GBSA sampling

import matplotlib.pyplot as plt
import numpy as np

from serial_sample import load_freesolv

legend, db = load_freesolv()

print(legend)

# let's plot how well the explicit solvent predictions match up with the experimental measurements:
x = np.array([float(entry[5]) for entry in db])
y = np.array([float(entry[3]) for entry in db])
plt.scatter(x, y, s=4)

line = range(int(min(x)), int(max(x)))
plt.plot(line, line, '--')
plt.xlabel("Predicted (Mobley)")
plt.ylabel("Measured")

plt.savefig("freesolv_prediction_vs_measured.pdf")
plt.close()
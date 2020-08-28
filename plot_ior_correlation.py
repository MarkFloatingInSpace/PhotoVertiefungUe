import numpy as np
import matplotlib.pyplot as plt

ior_correlation = np.load('data2/ior_correlation.npy')

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(ior_correlation, vmin=-1, vmax=1, cmap='RdBu')
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(ior_correlation):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

labels = ['x0', 'y0', 'c', 'k1', 'k2', 'p1', 'p2']

ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)

plt.show()
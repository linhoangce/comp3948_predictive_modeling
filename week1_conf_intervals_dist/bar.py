import matplotlib.pyplot as plt

x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]

# plt.bar(x, energy, color='green')
# plt.xlabel('Energy Source')
# plt.ylabel('Energy Output (GJ)')
# plt.title('Energy output from various fuel source')
#
# plt.xticks(x, x)
# plt.show()

### Group Bar Plots

import numpy as np

NUM_MEANS = 5
NUM_GROUPS = 3
bc_means = [20, 35, 30, 35, 27]
alberta_means = [25, 32, 34, 20, 25]
saskatchewan_means = [18, 28, 32, 24, 31]

# generates indices from 0 to 4
idx = np.arange(NUM_MEANS - 1)
width = 0.25

plt.bar(idx, bc_means[:-1], width, label='BC')
plt.bar(idx+width, alberta_means[:-1], width, label='AB')
plt.bar(idx+width*2, saskatchewan_means[:-1], width, label='SK')
plt.title('Quarterly Revenue by Province')
plt.xticks(idx+(width*3/NUM_GROUPS), ('Q1', 'Q2', 'Q3', 'Q4'))
plt.legend(loc='best')
plt.show()
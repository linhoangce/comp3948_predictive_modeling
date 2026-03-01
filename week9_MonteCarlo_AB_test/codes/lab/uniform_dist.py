import numpy as np
import matplotlib.pyplot as plt

LOW = 0
HIGH = 20
SIZE = 1200
SIMS = 3

plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

for i in range(1, SIMS + 1):
    x = np.random.uniform(LOW, HIGH, SIZE)
    plt.subplot(1, 3, i)
    plt.hist(x, 6, density=True)
    plt.xlabel('X')
    plt.ylabel('Frequency')

plt.rcParams.update({"font.size": 40})
plt.show()
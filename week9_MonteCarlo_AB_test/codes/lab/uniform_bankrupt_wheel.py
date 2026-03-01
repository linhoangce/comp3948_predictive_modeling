import numpy as np
import matplotlib.pyplot as plt

SIZE = 100
LOW = 0
HIGH = 24
SIMS = 5

# bankrupt position on the wheel
BANKRUPT = [11, 23]
avgs = []

for i in range(SIMS):
    spins = np.random.randint(LOW, HIGH, SIZE)

    # count bankrupts
    bankrupts = np.sum(np.isin(spins, BANKRUPT))
    print(f"Number of bankrupts: {bankrupts}")
    print(f"Bankrupt rate: {bankrupts/SIZE:.2f}")
    avgs.append(bankrupts)

plt.hist(avgs)
plt.xlabel('Number of Bankrupts')
plt.ylabel('Frequency')
plt.title("Distribution over 5 simulations")
plt.show()
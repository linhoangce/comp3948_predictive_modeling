from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

shape = 1
scale = 2

Gamma = gamma(a=shape, scale=scale)

x = np.arange(1, 20)

plt.plot(Gamma.pdf(x), x)
plt.xlabel("Duration of visit")
plt.ylabel("Number of visitors")
plt.show()
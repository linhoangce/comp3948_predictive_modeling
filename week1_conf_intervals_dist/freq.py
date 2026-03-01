import matplotlib.pyplot as plt
import numpy as np

def equation(x):
    return 3 * np.array(x) + 2

x = [-2, -1, 0, 1, 2, 3, 4]
y = equation(x)

plt.plot(x, y)
plt.axhline(0, color='black', )
plt.axvline(0, color='black')
# plt.xticks([]); plt.yticks([])
plt.ylim(-5, 15)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
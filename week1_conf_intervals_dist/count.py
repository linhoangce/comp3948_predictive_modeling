# import math
# import numpy as np
#
# x = np.array([2.5, 2.1, 1.8, 2.1, 2.3, 2.2, 2.2, 2.0, 1.9])
# # std = x.std()
# # print(std)
# #
# # x_mean = x.mean()
# # print(x_mean)
#
# def mean(x):
#     return np.sum(i for i in x) / len(x)
#
# # print(mean(x))
# x_mean = (2.5+ 2.1+ 1.8+ 2.1+ 2.3+ 2.2+ 2.2+ 2.0+ 1.9) / 9
# # print(mean(x) == x.mean())
#
# def std(x, x_mean):
#     return np.sqrt(1/(len(x)-1) * np.array([(i - x_mean)**2  for i in x]).sum())
#
# std = std(x, x_mean)
# print(std)
# print(x.std())
#
# x_mean = 2.12222222
# s = 0.1081851
# SE = s / math.sqrt(9)
# z = 2.306
#
# CI_upper = x_mean + z * SE
# CI_lower = x_mean - z * SE
# print(CI_upper)
# print(CI_lower)

L = [4, 3, 2, 1]
for i in range(len(L)):
    L.insert(i, L.pop())
print(L)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x = np.arange(1, 101, 1)
f = open('example.txt')
y = []

# print(f)

for each in f:
    y.append(float(each.split('\n')[0]))
    # print(each.split('\n')[0])

# print(y)

y_1 = []
for each in y:
    y_1.append(0.8*(each)/(0.6045))
# plt.plot(x, y_1)
# plt.show()

for each in y_1:
    print(each)

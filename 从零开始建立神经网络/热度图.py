import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./权重的十个特征矩阵/权重特征矩阵 0 epoch20', dtype=np.float32, delimiter=' ')
data_re = data.reshape(28, 28)

plt.imshow(data_re, cmap='inferno')
plt.show()
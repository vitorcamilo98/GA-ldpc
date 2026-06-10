import matplotlib.pyplot as plt

import numpy as np

A = np.load("matrizes/A_aleatoria_2.npy")

print(A)
plt.imshow(A, cmap="gray_r")
plt.title("Matriz A")
plt.show()
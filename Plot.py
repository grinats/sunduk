import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return t ** 2 * np.exp(-t ** 2)


t = np.linspace(-4, 4, 100)  # 100 точек между -4 и 4
y = f(t)

plt.plot(t, y)
plt.show()

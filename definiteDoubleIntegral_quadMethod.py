import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import dblquad


'''Пример вычисления объема правильной четырехугольной пирамиды со стороной a=2 и высотой h=1:
                    V = (1/3) * h * a**2
Уравнение поверхности пирамиды имеет вид z = 1 - |x| - |y|,
а область интегрирования ограничена снизу и сверху кривыми g(x) = |x| - 1 и h(x) = 1 - |x|.
Тогда ее объем можно найти следующим образом:
'''


def func_f(y, x):
    return 1 - np.abs(x) - np.abs(y)


def func_g(x):
    return np.abs(x) - 1


def func_h(x):
    return 1 - np.abs(x)


h = 1
a = np.sqrt(2)

result = dblquad(func_f, -1, 1, func_g, func_h)
print('\n\nЗначение интеграла (объем пирамиды):', result[0], '\nАбсолютная погрешность:', result[1])


# Изображение подинтегральной функции:
fig = plt.figure()
ax = Axes3D(fig)
x = [-1, 0, 0, -1, 0, 0, 1, 0, 1, 0]
y = [0, 0, -1, 0, 1, 0, 0, 1, 0, -1]
z = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
ax.plot(x, y, z, linewidth=3)
plt.show()

from scipy.integrate import quad
''' 
-- The function quad is provided to integrate a function of one variable between two points.
Help on function quad in module scipy.integrate.quadpack: help(scipy.integrate.quad)
'''
import numpy as np
# -- for exp() and sin() library functions.
import matplotlib.pyplot as plt


def f(x):
    return np.exp(-x)*np.sin(x)


result = quad(f, -np.pi, np.pi, full_output=True)
# Аргументы quad(): ссылка на функцию f, нижний (a) и верхний (b) пределы интегрирования.
# При задании опции full_output=True функция quad возвращает словарь с дополнительной информацией.
print('\n\nЗначение интеграла:', result[0], '\nАбсолютная погрешность:', result[1])
print('\nДоп параметры:\n', result)

# Построение графика подинтегральной функции:
t = np.linspace(-np.pi, np.pi, 100)  # 100 точек между -pi и pi
y = f(t)
plt.plot(t, y)
plt.show()

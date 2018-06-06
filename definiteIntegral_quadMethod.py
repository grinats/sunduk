from scipy.integrate import quad
''' 
-- The function quad is provided to integrate a function of one variable between two points.
Help on function quad in module scipy.integrate.quadpack: help(scipy.integrate.quad) or scipy.integrate.quad_explain()
'''
import numpy as np  # -- for exp() and sin() library functions.
import matplotlib.pyplot as plt  # -- for plot generation.
from matplotlib.patches import Polygon  # -- for plot generation.


def f(x):
    #return np.exp(-x)*np.sin(x)
    return np.sin(x)
    #return np.exp(-x)


def get_i():
    return -np.cos(-np.pi) + np.cos(np.pi)


a, b = -np.pi, np.pi  # integral limits
result = quad(f, a, b, full_output=True)
# Аргументы quad(): ссылка на функцию f, нижний (a) и верхний (b) пределы интегрирования.
# При задании опции full_output=True функция quad возвращает словарь с дополнительной информацией.
print('\n\nЗначение интеграла:', result[0], '\nАбсолютная погрешность:', result[1])
print('\nReference value:', get_i())
# print('\nДоп параметры:\n', result)


# Integrand function plot:
x = np.linspace(a, b, 100)  # 100 points between a and b.
y = f(x)
fig, ax = plt.subplots()
plt.plot(x, y)

# Make the shaded region
ix = np.linspace(a, b)
iy = f(ix)
verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
ax.add_patch(poly)

# Add text and grid lines
res = str(result[0])
err = '%.1e' % result[1]
limits = '\n a = ' + '%.5f' % a + '    b = ' '%.5f' % b
plt.title(r'$\int_a^b f(x)\mathrm{d}x$ = ' + res + ',  Abs.error: ' + err + limits, size='large')
plt.figtext(0.9, 0.05, '$x$')
plt.figtext(0.1, 0.9, '$y$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.grid()

# Show plot
plt.show()

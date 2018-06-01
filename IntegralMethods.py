from matplotlib import mlab
import math
import random


def get_i():
    return math.e ** 1 - math.e ** 0


def method_of_rectangles(func, mim_lim, max_lim, delta):
    def integrate(func, mim_lim, max_lim, n):
        integral = 0.0
        step = (max_lim - mim_lim) / n
        for x in mlab.frange(mim_lim, max_lim-step, step):
            integral += step * func(x + step / 2)
        return integral

    d, n = 1, 1
    while math.fabs(d) > delta:
        d = (integrate(func, mim_lim, max_lim, n * 2) - integrate(func, mim_lim, max_lim, n)) / 3
        n *= 2

    print('Method of rectangles:')
    print(' '.join([
        '\t',
        str(n),
        str(math.fabs(integrate(func, mim_lim, max_lim, n))),
        str(math.fabs(integrate(func, mim_lim, max_lim, n)) + d)]))


def trapezium_method(func, mim_lim, max_lim, delta):
    def integrate(func, mim_lim, max_lim, n):
        integral = 0.0
        step = (max_lim - mim_lim) / n
        for x in mlab.frange(mim_lim, max_lim-step, step):
            integral += step*(func(x) + func(x + step)) / 2
        return integral

    d, n = 1, 1
    while math.fabs(d) > delta:
        d = (integrate(func, mim_lim, max_lim, n * 2) - integrate(func, mim_lim, max_lim, n)) / 3
        n *= 2

    print('Trapezium method:')
    print(' '.join([
        '\t',
        str(n),
        str(math.fabs(integrate(func, mim_lim, max_lim, n))),
        str(math.fabs(integrate(func, mim_lim, max_lim, n)) + d)]))


def simpson_method(func, mim_lim, max_lim, delta):
    def integrate(func, mim_lim, max_lim, n):
        integral = 0.0
        step = (max_lim - mim_lim) / n
        for x in mlab.frange(mim_lim + step / 2, max_lim - step / 2, step):
            integral += step / 6 * (func(x - step / 2) + 4 * func(x) + func(x + step / 2))
        return integral

    d, n = 1, 1
    while math.fabs(d) > delta:
        d = (integrate(func, mim_lim, max_lim, n * 2) - integrate(func, mim_lim, max_lim, n)) / 15
        n *= 2

    print('Simpson\'s method:')
    print(' '.join([
        '\t',
        str(n),
        str(math.fabs(integrate(func, mim_lim, max_lim, n))),
        str(math.fabs(integrate(func, mim_lim, max_lim, n)) + d)]))


def monte_karlo_method(func, n):
    in_d, out_d = 0., 0.
    for i in range(n):
        x, y = random.uniform(0, 1), random.uniform(0, 3)
        if y < func(x): in_d += 1

    print('Monte-Karlo method:')
    print('\t' + str(n) + ' ' + str(math.fabs(in_d/n * 3)))


method_of_rectangles(lambda x: math.e ** x, 0.0, 1.0, 0.001)
trapezium_method(lambda x: math.e ** x, 0.0, 1.0, 0.001)
simpson_method(lambda x: math.e ** x, 0.0, 1.0, 0.001)
monte_karlo_method(lambda x: math.e ** x, 100)
print('Reference value:', get_i())

import numpy as np


def squared(m, n): return all(len(row) == len(m) == len(n) for row in m)


A = np.array([[3, -2, 2], [-1, -1, 2], [0, -5, -1]])
b = np.array([-2, -4, -1])
print('\n Матрица |A|:\n', A, '\n\nВектор свободных членов b:\n', b)

if A.dtype != 'float64' and A.dtype != 'int32':
    print(A.dtype, 'is not equal to float64 or int32.')
    exit(0)

if not squared(A, b):
    print('\nError: Size mismatch detected or input a is not square.')
else:
    try:
        d = np.linalg.det(A)
        x = np.linalg.solve(A, b)
        print('\nОпределитель матрицы |A|:\n', d)
        print('\n\nРешение СЛАУ |A|x=b:\n', 'x =', x)
        print('\nПроверка решения |A|х - b = 0:\n', np.dot(A, x) - b)
    except ValueError:
        print('\nValueError: size mismatch detected or input a is not square.')
    except np.linalg.linalg.LinAlgError:
        d = np.linalg.det(A)
        print('\nLinAlgError: the matrix is singular.')
        print('Определитель матрицы |A|:', d)
    except NotImplementedError:
        print('\nNotImplementedError: transposed is True and input a is a complex matrix.')
    '''
    except np.linalg.linalg.LinAlgWarning:
        print('\nLinAlgWarning: an ill-conditioned input a is detected.')
    '''

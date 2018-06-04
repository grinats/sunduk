import numpy as np


def slae(A, b):
    squared = lambda A, b:  all(len(row) == len(A) == len(b) for row in A)
    if A.dtype != 'float64' and A.dtype != 'int32':
        print('\nError: Wrong data type detected.\nYou have to use data of float64 or int32 type to fill matrices.\n')
        return
    if not squared(A, b):
        print('\nError: Size mismatch detected or input |A| is not square matrix.')
    else:
        try:
            result = np.linalg.solve(A, b)
            return result
        except ValueError:
            print('\nValueError: size mismatch detected or input a is not square.')
        except np.linalg.linalg.LinAlgError:
            d = np.linalg.det(A)
            print('\nLinAlgError: the matrix is singular.')
            print('Определитель матрицы |A|:', d)
        except NotImplementedError:
            print('\nNotImplementedError: transposed is True and input a is a complex matrix.')


A = np.array([[-2, -2, 5], [-1, -1, 2], [0, -5, -1]])
b = np.array([-2, -4, -1])
x = slae(A, b)
d = np.linalg.det(A)
print('\nМатрица |A|:\n', A, '\n\nВектор свободных членов b:\n', b)
print('\nОпределитель матрицы |A|:\n', d)
print('\nРешение СЛАУ |A|x=b:\n', 'x =', x)
print('\nПроверка решения |A|х - b = 0:\n', np.dot(A, x) - b)

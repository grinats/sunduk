import numpy as np


def slae(A, b):

    def squared(m, n):
        return all(len(row) == len(m) == len(n) for row in m)

    if not squared(A, b):
        print('\nError: Size mismatch detected or input |A| is not square matrix.')
        return None

    if A.dtype != 'float64' and A.dtype != 'int32':
        print('\nError: Wrong data type detected.\nYou have to use input data of float64 or int32 type.\n')
        return None

    try:
        result = np.linalg.solve(A, b)
        return result
    except ValueError:
        print('\nValueError: Size mismatch detected or input |A| is not square.')
        return None
    except np.linalg.linalg.LinAlgError:
        determinant = np.linalg.det(A)
        print('\nLinAlgError: The matrix |A| is singular.\nDeterminant |A| is:', determinant)
        return None
    except NotImplementedError:
        print('\nNotImplementedError: Transposed is True and input |A| is a complex matrix.')
        return None


# Input:
#  A = np.array([[2, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]], dtype=float)
#  b = np.array([-2, -4, -1, -2], dtype=float)

#  Random matrix testing.
k = 1  # Matrix elements values are Gauss distributed within the given interval: -k <= x < k
n = 3  # Dimension of matrix |A|(n x n) and vector b(n).
A = np.random.uniform(-k, k, size=(n, n))
b = np.random.uniform(-k, k, size=(n,))

# Solve:
x = slae(A, b)

# Output:
if x is not None:
    d = np.linalg.det(A)
    print('\nМатрица |A|:\n', A, '\n\nВектор свободных членов b:\n', b)
    print('\nОпределитель матрицы |A|:\n', d)
    print('\nРешение СЛАУ |A|x=b:\n', 'x =', x)
    print('\nПроверка решения |A|х - b = 0:\n', np.dot(A, x) - b)

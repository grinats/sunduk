import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def slae(A, b):
    """
    Parameters:
        A - Coefficient matrix. A must be square and of full-rank,
            i.e., all rows (or, equivalently, columns) must be linearly independent.
            If either is not true, use lstsq for the least-squares best “solution” of the system/equation.
        b - Ordinate or “dependent variable” values.
    Returns:
        x - Solution to the system a x = b. Returned shape is identical to b.
    """

    def squared(m, n):
        """
            Check if matrix is squared and has non zero dimension.
            Parameters:
                m - Coefficient matrix.
                n - Ordinate or “dependent variable” values.
            Returns:
                True - if all requirements met. Otherwise returns None.
            """
        return all(len(row) == len(m) == len(n) for row in m) and len(n) != 0

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


def drawPlot(A, x, b):  # Plot the 1, 2 or 3 linear functions graph.
    """
    Parameters:
        A - Coefficient matrix.
        x - Solution to the system A x = b.
        b - Ordinate or “dependent variable” values.
    Returns:
         Visualise solution to the system A x = b in case of matrix dimension is greater than zero and less then four.
    """
    n = len(b)
    if n <= 2:
        t = np.linspace(x[0] - 0.2, x[0] + 0.2, 100)
        if n == 1:  # One linear equation
            f1 = A[0, 0]*t - b[0]
            plt.plot(t, f1)
            plt.plot(x[0], 0, 'bo')
            plt.title('%.3f' % A[0, 0] + ' * x - (' + '%.3f' % b[0] + ') = 0,\tх = ' + '%.3f' % x[0], size='large')
        elif n == 2:  # System of two linear equations
            f1 = (b[0] - A[0, 0] * t)/A[0, 1]
            f2 = (b[1] - A[1, 0] * t)/A[1, 1]
            plt.plot(t, f1)
            plt.plot(t, f2)
            plt.plot(x[0], x[1], 'bo')
            title_f1 = '%.3f' % A[0, 0] + ' * x +' + ' ( %.3f' % A[0, 1] + ' * y) - (' + '%.3f' % b[0] + ') = 0,     '
            title_x = 'х = ' + '%.3f' % x[0]
            title_f2 = '%.3f' % A[1, 0] + ' * x +' + ' ( %.3f' % A[1, 1] + ' * y) - (' + '%.3f' % b[1] + ') = 0,     '
            title_y = 'y = ' + '%.3f' % x[1]
            plt.title(title_f1 + title_x + '\n' + title_f2 + title_y, size='large')
        plt.figtext(0.9, 0.03, '$x$')
        plt.figtext(0.1, 0.9, '$y$')
        ax = plt.gca()
        ax.grid()
        plt.figure(1)
        plt.show()
    elif n == 3:  # System of three linear equations
        def f1(x, y):
            return (b[0] - A[0, 0] * x - A[0, 1] * y) / A[0, 2]

        def f2(x, y):
            return (b[1] - A[1, 0] * x - A[1, 1] * y) / A[1, 2]

        def f3(x, y):
            return (b[2] - A[2, 0] * x - A[2, 1] * y) / A[2, 2]

        fig = plt.figure(figsize=(8, 8))  # Figure size in inches.
        ax = fig.gca(projection='3d')
        xPlot = np.arange(x[0] - 1, x[0] + 1, 0.05)
        yPlot = np.arange(x[1] - 1, x[1] + 1, 0.05)
        X, Y = np.meshgrid(xPlot, yPlot)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title_f1 = '%.3f' % A[0, 0] + ' * x +' + ' ( %.3f' % A[0, 1] + ') * y +' + \
                   ' ( %.3f' % A[0, 2] + ') * z - (' + '%.3f' % b[0] + ') = 0,'
        title_x = 'х = ' + '%.3f' % x[0]
        title_f2 = '%.3f' % A[1, 0] + ' * x +' + ' ( %.3f' % A[1, 1] + ') * y +' + \
                   ' ( %.3f' % A[1, 2] + ') * z - (' + '%.3f' % b[1] + ') = 0,'
        title_y = 'y = ' + '%.3f' % x[1]
        title_f3 = '%.3f' % A[2, 0] + ' * x +' + ' ( %.3f' % A[2, 1] + ') * y +' + \
                   ' ( %.3f' % A[2, 2] + ') * z - (' + '%.3f' % b[2] + ') = 0,'
        title_z = 'z = ' + '%.3f' % x[2]
        plt.title(title_f1 + '   ' + title_x + '\n' + title_f2 + '   ' + title_y + '\n' + title_f3 + '   ' + title_z, size='large')

        # Plane 1 plot
        z1 = np.array([f1(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z1 = z1.reshape(X.shape)
        ax.plot_surface(X, Y, Z1, rstride=8, cstride=8, alpha=0.1)

        # Plane 2 plot
        z2 = np.array([f2(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z2 = z2.reshape(X.shape)
        ax.plot_surface(X, Y, Z2, rstride=8, cstride=8, alpha=0.1)

        # Plane 3 plot
        z3 = np.array([f3(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z3 = z3.reshape(X.shape)
        ax.plot_surface(X, Y, Z3, rstride=8, cstride=8, alpha=0.1)

        # Intersection point plot
        ax.plot([x[0]], [x[1]], [x[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)

        plt.show()


# Input:
# A = np.array([[2, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]], dtype=float)
# b = np.array([-2, -4, -1, -2], dtype=float)
# A = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
# b = np.array([1, 1, 1], dtype=float)

#  Random matrix testing.
n = 3  # Dimension of matrix |A|(n x n) and vector b(n).
k = 1  # Matrix elements values are Gauss distributed within the given interval: -k <= x < k
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
    drawPlot(A, x, b)

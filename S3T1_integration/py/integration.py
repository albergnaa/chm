import numpy as np
import math

from utils.integrate_collection import Monome


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """
    assert alpha * beta == 0, \
        f'at least one of alpha ({alpha}) or beta ({beta}) should be 0'

    if alpha == 0 and beta != 0:
        listOfMoments = [((b - xr) ** (1 - beta) - (b - xl) ** (1 - beta)) / (1 - beta) * (-1)]
        if max_s > 0:
            for i in range(1, max_s + 1):
                listOfMoments.append(((b - xr) ** (i + 1 - beta) - (b - xl) ** (i + 1 - beta)) / (i + 1 - beta) * (-1))
                for j in range(i - 1, -1, -1):
                    coefficient = math.factorial(i) / math.factorial(i - j) / math.factorial(j) * (-1) ** (
                            j + 1) * b ** (i - j)
                    listOfMoments[i] += listOfMoments[j] * coefficient
                listOfMoments[i] *= (-1) ** i
        return listOfMoments

    if alpha != 0 and beta == 0:
        listOfMoments = [((xr - a) ** (1 - alpha) - (xl - a) ** (1 - alpha)) / (1 - alpha)]
        if max_s > 0:
            for i in range(1, max_s + 1):
                listOfMoments.append(((xr - a) ** (i + 1 - alpha) - (xl - a) ** (i + 1 - alpha)) / (i + 1 - alpha))
                for j in range(i - 1, -1, -1):
                    coefficient = math.factorial(i) / math.factorial(i - j) / math.factorial(j) * a ** (i - j)
                    if (i - j) % 2 == 0:
                        coefficient *= -1
                    listOfMoments[i] += listOfMoments[j] * coefficient
        return listOfMoments

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]


def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """
    return -math.log(abs((s2 - s1) / (s1 - s0))) / math.log(L)


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """
    n = len(xs)
    listOfMoments = moments(n - 1, x0, x1, **kwargs)

    X = [[0] * len(xs) for i in range(len(xs))]
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            X[i][j] = xs[j] ** i

    vectorOfMoments = np.array(listOfMoments)
    A = np.linalg.solve(X, vectorOfMoments)

    sum = 0
    for i in range(0, n):
        sum += A[i] * func(xs[i])
    return sum


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    listOfMoments = moments(2 * n - 1, x0, x1, **kwargs)

    matrixOfMoments = [[0] * n for i in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            matrixOfMoments[i][j] = listOfMoments[j + i]
    vectorOfMomentsFromNTo2N = np.array(listOfMoments[n:2 * n])

    a = np.linalg.solve(matrixOfMoments, -vectorOfMomentsFromNTo2N)
    a = a[::-1]
    a = np.insert(a, 0, 1)
    xs = np.roots(a)

    X = [[0] * len(xs) for i in range(len(xs))]
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            X[i][j] = xs[j] ** i

    listOfMomentsFrom0ToN = listOfMoments[0:n]
    A = np.linalg.solve(X, listOfMomentsFrom0ToN)
    sum = 0
    for i in range(0, n):
        sum += A[i] * func(xs[i])
    return sum


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    nodes = np.linspace(x0, x1, n_intervals + 1)
    sum = 0
    for i in range(n_intervals):
        sum += quad(func, nodes[i], nodes[i + 1], np.linspace(nodes[i], nodes[i + 1], n_nodes), **kwargs)
    return sum


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    nodes = 3
    L = 2
    h0 = x1 - x0
    r2 = 1
    while r2 > tol:
        listOfResult = []
        h = []
        for i in range(3):
            h.append(h0 / L ** i)
            n = math.floor((x1 - x0) / h[i])
            listOfResult.append(composite_quad(func, x0, x1, n, nodes))
        m = aitken(listOfResult[0], listOfResult[1], listOfResult[2], L)
        r1, r2 = runge(listOfResult[1], listOfResult[2], m, L)
        h0 = h[2] * math.pow(tol / abs(r2), 1 / m)
    return listOfResult[2], max(r2, tol)


p = Monome(1)
x0, x1 = 0, 1
y0 = p[x0, x1]
xs1 = np.linspace(x0, x1, 1)
xs2 = [(x0+x1)/2]
Y1 = quad(p, x0, x1, xs1)
Y2 = quad(p, x0, x1, xs2)
print("узел:", xs1, ", точное: ", y0, ", вычисленное:", Y1)
print("узел:", xs2, ", точное: ", y0, ", вычисленное:", Y2)
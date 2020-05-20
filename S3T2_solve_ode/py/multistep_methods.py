import numpy as np
from S3T2_solve_ode.py.one_step_methods import OneStepMethod


#  coefficients for Adams methods
adams_coeffs = {
    1: [1],
    2: [-1 / 2, 3 / 2],
    3: [5 / 12, -4 / 3, 23 / 12],
    4: [-3 / 8, 37 / 24, -59 / 24, 55 / 24],
    5: [251 / 720, -637 / 360, 109 / 30, -1387 / 360, 1901 / 720]
}


def adams(func, y_start, T, coeffs, one_step_method: OneStepMethod):
    """
    T: list of timestamps
    coeffs: list of coefficients
    one_step_method: method for initial steps
    return list of t (same as T), list of y
    """

    ys = [y_start]
    n = len(coeffs)
    F = []
    for i, t in enumerate(T[0:n]):
        y = ys[-1]
        y1 = one_step_method.step(func, t, y, T[i + 1] - t)
        ys.append(y1)
        F.append(func(T[i + 1], y1))

    for i, t in enumerate(T[n:-1]):
        temp = 0
        for j in range(0, n):
            temp += coeffs[j] * F[i + j]
        ys.append(ys[-1] + (T[n + i + 1] - t) * temp)
        F.append(func(T[n + i + 1], ys[-1]))
    return T, ys


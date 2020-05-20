import enum
import numpy as np

from S3T2_solve_ode.py.one_step_methods import OneStepMethod


class AdaptType(enum.Enum):
    RUNGE = 0
    EMBEDDED = 1


def fix_step_integration(method: OneStepMethod, func, y_start, ts):
    """
    performs fix-step integration using one-step method
    ts: array of timestamps
    return: list of t's, list of y's
    """
    ys = [y_start]

    for i, t in enumerate(ts[:-1]):
        y = ys[-1]

        y1 = method.step(func, t, y, ts[i + 1] - t)
        ys.append(y1)

    return ts, ys


def adaptive_step_integration(method: OneStepMethod, func, y_start, t_span,
                              adapt_type: AdaptType,
                              atol, rtol):
    """
    performs adaptive-step integration using one-step method
    t_span: (t0, t1)
    adapt_type: Runge or Embedded
    tolerances control the error:
        err <= atol
        err <= |y| * rtol
    return: list of t's, list of y's
    """
    y = y_start
    t, t_end = t_span
    ys = [y]
    ts = [t]
    if method.name == 'Euler (explicit)':
        p = 1
    else:
        p = method.p
    delta = pow((1 / max(np.abs(t), np.abs(t_end))), p + 1) + pow(np.linalg.norm(func(t, y)), p + 1)
    dt = pow((atol / delta), (1 / (p + 1)))
    while ts[-1] < t_end:
        dt = min(dt, np.abs(t_end - ts[-1]))
        tn = ts[-1]
        yn = ys[-1]
        if adapt_type == AdaptType.RUNGE:
            y1 = method.step(func, tn, yn, dt)
            y2 = method.step(func, tn, yn, dt / 2)
            y3 = method.step(func, tn + dt / 2, y2, dt / 2)
            err = (y3 - y1) / (pow(2, p) - 1)
        if adapt_type == AdaptType.EMBEDDED:
            y3, err = method.embedded_step(func, tn, yn, dt)
        Nerr = np.linalg.norm(err)
        adapt_tol = rtol * np.maximum(np.linalg.norm(y3), np.linalg.norm(y)) + atol

        # if Nerr > adapt_tol * pow(2, p):
        #     dt = dt / 2
        # else:
        #     if Nerr > adapt_tol:
        #         dt = dt / 2
        #         ts.append(tn + dt)
        #         ys.append(y3)
        #     else:
        #         if Nerr >= adapt_tol / pow(2, p + 1):
        #             ts.append(tn + dt)
        #             # должно браться y1
        #             ys.append(y3)
        #         else:
        #             # обычно еще дана максимальная допустимая длина шага, берется min
        #             dt = 2 * dt
        #             ts.append(tn + dt)
        #             # должно браться y1
        #             ys.append(y3)

        if Nerr > adapt_tol:
            dt = dt / 2
        else:
            if Nerr > adapt_tol / pow(2, p + 1):
                ys.append(y3)
                ts.append(tn + dt)
            else:
                ys.append(y3)
                ts.append(tn + dt)
                # dt = dt * 0.5 * pow(adapt_tol / Nerr,  1 / p)
                # dt = dt * pow(adapt_tol / Nerr,  1 / p)
                # dt = dt * pow(adapt_tol / Nerr,  1 / (p + 1))
                # dt = dt * 0.5 * pow(adapt_tol / Nerr, 1 / (p + 1))
                dt = dt * 0.5 * pow(adapt_tol / (Nerr * pow(2, p)), 1 / (p + 1))
                # print('!')
        # print(dt)
    return ts, ys

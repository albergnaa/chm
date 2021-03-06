import pytest
import numpy as np
import matplotlib.pyplot as plt

from S2T1_optimization.py import optimization

default_student = 0


@pytest.mark.parametrize("student", [default_student])  # student num
@pytest.mark.parametrize("n_dim", [2, 3])               # dimension number
def test_optimization(student, n_dim):
    print(f'running optimization test for student #{student}')
    N = student

    A = np.array([[4,  1,      1],
                  [1,  6+.2*N, -1],
                  [1,  -1,      8+.2*N]],
                 dtype='float')[:n_dim, :n_dim]
    b = np.array([1, -2, 3], dtype='float').reshape(-1, 1)[:n_dim, :]
    x0 = np.array([0, 0, 0], dtype='float').reshape(-1, 1)[:n_dim, :]

    eps_y = 1e-6
    eps_x = 1e-3

    methods = ['mngs', 'mps']
    styles = ['mo-', 'b.:']
    plt.figure()
    plt.title(f'Результаты для размерности {n_dim}')
    plt.xlabel('номер итерации')
    plt.ylabel('точность')
    for i, method in enumerate(methods):
        optimization.func.calls = 0
        X, Y = getattr(optimization, method)(A, b, x0, eps_y)

        x1 = np.linalg.solve(A, -b)
        y1 = (1/2 * x1.T @ A @ x1 + b.T @ x1).item()

        assert np.equal(x0, X[0]).all(), 'X should start with initial point'
        assert np.linalg.norm(x1 - X[-1]) < eps_x, 'last X should be close enough to the optimum'
        assert np.linalg.norm(y1 - Y[-1]) < eps_y, 'last Y should be close enough to the optimum'
        assert optimization.func.calls == len(Y), f'function was called {optimization.func.calls} times, ' \
                                                  f'but there is {len(Y)} point in the output'

        plt.plot(-np.log10([y - y1 for y in Y]), styles[i])
    plt.legend(methods)
    plt.show()

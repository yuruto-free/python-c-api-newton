import numpy as np

def objective_function(vec):
    """
    Parameters
    ----------
    vec : numpy.ndarray
        vector of x
        vec[i]: x_{i}

    Returns
    -------
    fv : numpy.ndarray
        vector of f(vec)
    """
    x, y = vec

    fv = np.array([
            x + 2 *     y + 1.0,
        x * x + 2 * y * y - 3.0,
    ])

    return fv

def calc_Jacobian_matrix(vec, delta=1e-3):
    """
    Parameters
    ----------
    vec : numpy.ndarray
        vector of x
        vec[i]: x_{i}
    delta : double
        fixed difference value

    Returns
    -------
    Jf : numpy.ndarray
        Jacobian matrix
        Jf[i, j] = \dfrac{\partial f_{i}}{\partial x_{j}}
    """
    ndim = vec.size
    Jf = np.empty((ndim, ndim), dtype=np.float64)

    for idx in np.arange(ndim):
        val = vec[idx]
        # calculate f(x + delta)
        vec[idx] = val + delta
        right_f= objective_function(vec)
        # calculate f(x - delta)
        vec[idx] = val - delta
        left_f = objective_function(vec)
        # calculate 0.5 * (f(x + delta) - f(x - delta)) / delta
        diffs = 0.5 * (right_f - left_f) / delta
        Jf[:, idx] = diffs
        vec[idx] = val

    return Jf

def newton_method(vec, max_iter, tol=1e-10, delta=1e-3):
    """
    Parameters
    ----------
    vec : numpy.ndarray
        vector of x
        vec[i]: x_{i}
    max_iter : int
        maximum iteration
    tol : double
        relative tolerance

    Returns
    -------
    hat : numpy.ndarray
        vector of \hat{x}
        hat[i]: \hat{x}_{i}
    is_convergent : boolean
        convergent status
            True:  convergent
            False: divergence
    """
    hat = np.copy(vec)
    is_convergent = False

    for _ in np.arange(max_iter):
        # convergence test
        if np.linalg.norm(hat) < tol:
            is_convergent = True
            break

        # Step1: calculate Jacobian matrix
        Jf = calc_Jacobian_matrix(hat, delta=delta)
        # Step2: calculate function value
        fv = objective_function(hat)
        # Step3: calculate delta_v
        delta_v = np.linalg.solve(Jf, -fv)
        # Step4: update vec
        hat += delta_v

    return [hat, is_convergent]

if __name__ == '__main__':
    max_iter = 1000
    tol = 1e-10

    # ===============
    # solve pattern 1
    # ===============
    print('Solve pattern 1')
    vec = np.array([2.0, 1.0])
    exact_vec = np.array([1.0, -1.0])
    print('[init value] x: {:.5f}, y: {:.5f}'.format(*vec))
    hat, _ = newton_method(vec, max_iter, tol=tol)
    err = np.linalg.norm(hat - exact_vec)
    print('[estimated]  x: {:.5f}, y: {:.5f} ({:.5e})'.format(*hat, err))
    print('')

    # ===============
    # solve pattern 2
    # ===============
    print('Solve pattern 2')
    vec = np.array([-1.0, 1.0])
    exact_vec = np.array([-5.0/3.0, 1.0/3.0])
    print('[init value] x: {:.5f}, y: {:.5f}'.format(*vec))
    hat, _ = newton_method(vec, max_iter, tol=tol)
    err = np.linalg.norm(hat - exact_vec)
    print('[estimated]  x: {:.5f}, y: {:.5f} ({:.5e})'.format(*hat, err))

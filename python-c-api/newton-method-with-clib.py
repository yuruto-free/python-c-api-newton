import numpy as np
import wrapper_newtonlib as newtonlib

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

    # | x     + 2y     = -1          x     + 2y     + 1 = 0
    # |                         <=>
    # | x^{2} + 2y^{2} = 3           x^{2} + 2y^{2} - 3 = 0
    x, y = vec

    fv = np.array([
            x + 2 *     y + 1.0,
        x * x + 2 * y * y - 3.0,
    ])

    return fv

if __name__ == '__main__':
    max_iter = 1000
    tol = 1e-10
    # set objective function
    newtonlib.set_objective_function(objective_function)

    # ===============
    # solve pattern 1
    # ===============
    print('Solve pattern 1')
    vec = np.array([2.0, 1.0])
    exact_vec = np.array([1.0, -1.0])
    print('[init value] x: {:.5f}, y: {:.5f}'.format(*vec))
    hat, _ = newtonlib.newton_method(vec, max_iter, tol=tol)
    err = np.linalg.norm(hat - exact_vec)
    print('[estimated]  x: {:.5f}, y: {:.5f} ({:.5f})'.format(*hat, err))
    print('')

    # ===============
    # solve pattern 2
    # ===============
    print('Solve pattern 2')
    vec = np.array([-1.0, 1.0])
    exact_vec = np.array([-5.0/3.0, 1.0/3.0])
    print('[init value] x: {:.5f}, y: {:.5f}'.format(*vec))
    hat, _ = newtonlib.newton_method(vec, max_iter, tol=tol)
    err = np.linalg.norm(hat - exact_vec)
    print('x: {:.5f}, y: {:.5f} ({:.5f})'.format(*hat, err))

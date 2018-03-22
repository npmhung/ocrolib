import numpy as np


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


def hessian_from_grad(gx, gy):
    gxy, gxx = np.gradient(gx)
    gyy, gyx = np.gradient(gy)
    shape = list(gx.shape)
    shape.extend([2, 2])
    hessian = np.zeros(shape)
    for y in range(gx.shape[0]):
        for x in range(gx.shape[1]):
            if gxy[y, x] != gyx[y, x]:
                print("Not ok")
            hessian[y, x, 0, 0] = gxx[y, x]
            hessian[y, x, 0, 1] = gxy[y, x]
            hessian[y, x, 1, 0] = gyx[y, x]
            hessian[y, x, 1, 1] = gyy[y, x]
    print(hessian[300, 300, :, :])
    return hessian

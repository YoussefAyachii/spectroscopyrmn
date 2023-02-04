"""functions"""

import numpy as np
import matplotlib.pyplot as plt

def sampling_grid(N, Tmin, Tmax):
    """Creating a sample grid"""
    
    samp_grid = np.zeros(N)
    n_vec = np.arange(1, N+1, 1)

    for i, n in enumerate(n_vec):
        samp_grid[i] = Tmin * np.exp(- (n - 1) * (np.log(Tmin/Tmax) / N-1))
    return samp_grid



def sampling_grid_regular(M, tmin, tmax):
    """Creating a regular sampling grid"""

    samp_grid = np.zeros(M)
    m_vec = np.arange(1, M+1, 1)

    for i, m in enumerate(m_vec):
        samp_grid[i] = tmin + ((m - 1)/ (M - 1)) * (tmax - tmin)
    return samp_grid


def build_K(t, T):
    """Building matrix K as : K m,n exp(- tm * Tn)"""
    N, M = len(T), len(t)
    K = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            K[m, n] = np.exp(- t[m] * T[n])
    return K


# Regularization term : R(x) = 1/2 * norm(||Dx||, order=2) **2

def build_D(x):
    """Building matrix D"""
    N = x.shape[0]
    D = np.identity(N)

    # add -1s in D
    for row_i in range(D.shape[0]):
        if row_i != 0:
            row_tmp = D[row_i, :]
            for i in range(len(row_tmp)):
                if row_tmp[i] == 1:
                    row_tmp[i - 1] = -1
            D[row_i, :] = row_tmp
    return D


def build_Dx(x):
    """Building matrix Dx"""
    D = build_D(x)
    return np.dot(D, x)


def ridge(Dx):
    """Conputes the square of the norm of Dx"""
    return np.square(np.linalg.norm(Dx, ord=2))

def ridge_deriv(x):
    """Computes the gradient of the L2 regularization
    with respect to x."""

    D = build_D(x)
    Dx = build_Dx(x)
    return np.dot(D.T, Dx)

def lasso(Dx):
    """Conputes the L1 of Dx"""
    return np.sum(np.abs(Dx))

def lasso_deriv(Dx):
    """Computes the gradient of the L1 regularization
    with respect to x."""
    Dx_deriv = np.zeros(len(Dx))
    for i, value in enumerate(Dx):
        assert value != 0  # derivative of 0 not defined
        if value > 0:
            Dx_deriv[i] = 1
        elif value < 0:
            Dx_deriv[i] = -1

    return Dx_deriv



# objective function:
def objective(x, y, K, lamb, fct_regu):
    Dx = build_Dx(x)
    Rx = 1/2 * fct_regu(Dx)
    return 1/2 * np.square(np.linalg.norm(y - np.dot(K, x), ord=2)) + lamb * Rx


# gradient of the objective function according to x
def gradient_f(x, y, K):
    """Computes the gradient of the objective function
    without the regularization term"""
    return - np.dot(K.T, y) + np.dot(np.dot(K.T, K), x)


# define optimal value of mu
def mu_opt(K):
    return 2/np.linalg.norm(K.T @ K)

# compute quadratic error between the original and the estimated signal
def quadratic_error(x, x_hat):
    # chack args
    assert len(x) == len(x_hat)
    # quadratic error x vs x_hat
    nom = np.square(np.linalg.norm(x_hat - x, ord=2))
    denom = np.square(np.linalg.norm(x, ord=2))
    return nom/denom



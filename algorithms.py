"""Optimization algorithms"""

import numpy as np
import matplotlib.pyplot as plt

from functions import (build_D, mu_opt, objective,
                       ridge, lasso, quadratic_error)


def descend(y, K, lamb, gradient, ips, regu="L2"):
    """gradient descend algorithm:
    to minimize the objective function"""

    # initialization
    x_hat_new = np.zeros(K.shape[1])
    objective_history = []
    stop_criteria = ips + 1
    
    mu_optimal = mu_opt(K)

    while stop_criteria > ips:  # we stop when stop_criteria >= ips
        # upload x_hat
        x_hat_new = x_hat_new - mu_optimal * gradient(x_hat_new, y, K)
        # upload stop_criteria
        # we want to stop when grdt is ~ 0
        stop_criteria = np.square(np.linalg.norm(gradient(x_hat_new, y, K), ord=2))

        # compute objective function value in each iteration
        objective_history.append(objective(x_hat_new, y, K, lamb, ridge))

    return x_hat_new, objective_history


def descend_prox(y, K, lamb, gradient, ips, regu):
    """proximal gradient descend algorithm:
    to minimizing the objective function with L2 reg."""

    # initialization
    x_n_new = np.ones(K.shape[1])

    stop_criteria = ips + 1
    
    objective_history = []

    D = build_D(x_n_new)
    
    mu_optimal = mu_opt(K)
    
    assert regu == "L2"
    while stop_criteria > ips:  # we stop when stop_criteria <= ips
        x_n = x_n_new
        # upload x_hat
        z_n_new = x_n_new - mu_optimal * gradient(x_n_new, y, K)
        x_n_new = np.linalg.solve(lamb * mu_optimal * np.dot(D.T, D) + np.identity(D.shape[0]), z_n_new)

        # upload stop_criteria
        nom = np.linalg.norm(x_n_new - x_n, ord=2)
        denom = np.linalg.norm(x_n, ord=2)
        stop_criteria = nom / denom

        # objective function value at each iteration
        objective_history.append(objective(x_n, y, K, lamb, ridge))

    return x_n_new, objective_history



def proxf_vec (cX, lamb):
    """compute prox f of values in vector"""

    # check args
    assert len(cX.shape) == 1  # only vector

    ncX = np.zeros(cX.shape)
    for i, cX_i in enumerate(cX):
        if np.abs(cX_i) < lamb:
            ncX_i = 0
        elif cX_i >= lamb:
            ncX_i = cX_i - lamb
        elif cX_i < -lamb:
            ncX_i = cX_i + lamb
        ncX[i] = ncX_i
    return ncX


# Use Forward Backward Primal Dual algorithm when using Lasso reg.


def proxf(cX, lamb):
    """compute prox f of values in matrix or vec"""
    # args check
    assert len(cX.shape) < 3  # cX must be only vector or matrix

    # if cX vec
    if len(cX.shape) == 1: 
        ncX = proxf_vec(cX, lamb)
    # if cX matrix
    elif len(cX.shape) == 2: # matrix
        ncX = np.zeros(cX.shape)
        for i in range(cX.shape[1]): # for each column
            tmp = cX[:, i]
            ncX[:, i] = proxf_vec(tmp, lamb)

    return ncX


def FB_primal_dual(y, K, lamb, tau, sigma, ips, regu):
    """forward backward primal dual algo to minimize
    the objective function"""

    # initialization
    x_n_new = np.ones(K.shape[1])
    v_n_new = np.ones(K.shape[1])

    stop_criteria = ips + 1  # initialization must be > ips
    
    objective_history = []

    D = build_D(x_n_new)    

    while stop_criteria > ips:  # we stop when stop_criteria <= ips
        # save x_n
        x_n = x_n_new
        
        # update x_n_new
        tmp_1 = x_n_new - tau * np.dot(D.T, v_n_new)
        x_n_new = np.linalg.solve(np.dot(K.T, K) + 1/tau * np.identity(K.shape[1]),
                                    1/2 * tmp_1 + np.dot(K.T, y))            
        tmp_2 = v_n_new + sigma * np.dot(D, x_n_new)
        v_n_new = tmp_2 - sigma * proxf(1/sigma * tmp_2, lamb=lamb)

        # upload stop_criteria
        nom = np.linalg.norm(x_n_new - x_n, ord=2)
        denom = np.linalg.norm(x_n, ord=2)
        stop_criteria = nom / denom

        if regu == "L1":
            #print("stop_criteria : ", stop_criteria)
            objective_history.append(objective(x_n, y, K, lamb, lasso))
        elif regu == "L2":
            objective_history.append(objective(x_n, y, K, lamb, ridge))

    return x_n_new, objective_history



# Find the best lambda in order to reduce the error

def find_best_lambda(x, y, K, lambdas, gradient, ips, algo, regu, output_dir=False):
    """Finding the best lambda minimizing the quadratic error
    between the original signal and the estimated signal"""
    quadratic_errors = np.zeros(len(lambdas))
    
    if algo == "descend_prox" :
        assert regu == "L2"  # only feasable if L2 regularization
        for i, l in enumerate(lambdas):
            x_hat, _ = descend_prox(y=y, K=K, lamb=l,
                            gradient=gradient, ips=ips,
                            regu=regu)
            quadratic_errors[i] = quadratic_error(x, x_hat)
        assert len(x) == len(x_hat)
        min_error = np.min(quadratic_errors)
        best_lambda = lambdas[np.argmin(quadratic_errors)]
    
    elif algo == "descend" :
        assert regu == "L2"  # only feasable if L2 regularization
        for i, l in enumerate(lambdas):
            x_hat, _ = descend(y=y, K=K, lamb=l,
                            gradient=gradient, ips=ips,
                            regu=regu)
            quadratic_errors[i] = quadratic_error(x, x_hat)
        assert len(x) == len(x_hat)
        min_error = np.min(quadratic_errors)
        best_lambda = lambdas[np.argmin(quadratic_errors)]

    elif algo == "FB_primal_dual":
        for i, l in enumerate(lambdas):
            x_hat, _ = FB_primal_dual(
                y, K, lamb=l, tau=0.1, sigma=0.1, ips=0.1, regu=regu)

            quadratic_errors[i] = quadratic_error(x, x_hat)
        assert len(x) == len(x_hat)
        assert len(lambdas) == len(quadratic_errors)
        min_error = np.min(quadratic_errors)
        best_lambda = lambdas[np.argmin(quadratic_errors)]

    if output_dir != False:
        fig, ax = plt.figure(), plt.axes()
        plt.plot(lambdas, quadratic_errors)
        plt.title("quadratic error between original signal and x_hat"
                  + " \n depending on lambda value")
        plt.scatter(best_lambda, min_error, color="red",
                    label=f"best lambda = {np.round(best_lambda, 2)}")
        plt.legend()
        plt.savefig(output_dir)

    return quadratic_errors
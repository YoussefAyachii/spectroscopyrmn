"""
Diffusion measurements by Nuclear Magnetic Resonance (NMR).
"""

import numpy as np
import matplotlib.pyplot as plt

from functions import (sampling_grid, sampling_grid_regular, build_K,
                       build_D, build_Dx, ridge, ridge_deriv, lasso,
                       lasso_deriv, objective, gradient_f, quadratic_error)
from algorithms import (descend, descend_prox, FB_primal_dual, find_best_lambda)

# load data: signal = measured intensity during DOSY experience
signal = np.loadtxt("inputs/signal_dosy.txt")
assert signal.shape[0] == 200 
print("signal.shape", signal.shape)


# sampling grid
Tmin = 1
Tmax = 1000
N = 200
T = sampling_grid(N, Tmin, Tmax)


# Display the original signal as function of T
fig, ax = plt.figure(), plt.axes()
plt.plot(T, signal)
ax.set_xscale('log')
plt.savefig("figures/original_signal.png")


# Build vector t using regular sampling grid strategy
M = 50
tmin = 0
tmax = 1.5
t = sampling_grid_regular(M, tmin, tmax)


# Build matrix K : Km,n = exp(− tm * Tn)
K = build_K(t=t, T=T)
print("K : \n", K.shape)


# Generate gaussian noise vector of mean = 0 and sd = 0.006.
mean = 0
sd = 0.006
w = np.random.normal(mean, sd, K.shape[0])
print("w shape  \n", w.shape)


# Build vector y where y = K * original_signal + w
y = np.dot(K, signal) + w
print("y shape  \n", y.shape)


# display y as function of t.
fig, ax = plt.figure(), plt.axes()
plt.scatter(t, y)
plt.yscale("log")
plt.title("y curve : \n y(t) = K * original_signal + w")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.savefig("figures/y(t).png")


# A. Objective function minimization with Ridge regularization 

# A.a. GRADIENT DESCEND ALGORITHM
x_hat_grdt_dscd_ridge, objective_history_grdt_dscd_ridge = descend(
    y=y, K=K, lamb=1, gradient=gradient_f, ips=0.001, regu="L2")


# plot objective_history_grdt_dscd_ridge 
fig, ax = plt.figure(), plt.axes()
plt.plot(objective_history_grdt_dscd_ridge)
plt.title("objective fct value in each iteration of the"
          + "gradient descend algo \n with ridge regularization")
plt.savefig("figures/objective_history_grdt_dscd_ridge.png")


# display x_hat_grdt_dscd_ridge
fig, ax = plt.figure(), plt.axes()
plt.plot(T, x_hat_grdt_dscd_ridge)
ax.set_xscale('log')
plt.title("estimated signal x_hat using \n"
          + "gradient descend algo with ridge regularization")
plt.savefig("figures/x_hat_grdt_dscd_ridge.png")


# compute quadratic error between orginal and estimated signal
quad_error = quadratic_error(x=signal, x_hat=x_hat_grdt_dscd_ridge)
print("gradient descend algo: \n "
      + "quadratique error x_bar vs x_hat_grdt_dscd_ridge \n : ", quad_error)


# find the best lambda minimizing the error
lambdas = np.linspace(0.0001, 100, 20)
quadratic_errors_ridge = find_best_lambda(
    x=signal, y=y, K=K, lambdas=lambdas, gradient=gradient_f,
    ips=0.01, algo="descend", regu="L2",
    output_dir="figures/quadratic_error_grdt_dscd_ridge.png")

min_error = np.min(quadratic_errors_ridge)
best_lamb = lambdas[np.argmin(quadratic_errors_ridge)]

print(f"grdt descend ridge case: min error ={min_error} for best lambda = {best_lamb}")


# A.b. PROXIMAL GRADIENT DESCEND ALGORITHM

# predict x_hat with ridge regularization using proximal gdt descend algo
x_hat_prox_ridge, objective_history_prox_ridge = descend_prox(
    y=y, K=K, lamb=1, gradient=gradient_f, ips=0.001, regu="L2")


# plot x_hat_prox_ridge 
fig, ax = plt.figure(), plt.axes()
plt.plot(objective_history_prox_ridge)
plt.title("objective fct value in each iteration of the"
          + "proximal gradient descend algo \n with ridge regularization")
plt.savefig("figures/objective_history_prox_ridge.png")


# display estimated signal x_hat_prox_ridge
fig, ax = plt.figure(), plt.axes()
plt.plot(T, x_hat_prox_ridge)
ax.set_xscale('log')
plt.title("estimated signal x_hat using \n"
          + "proximal gradient descend algorithm with ridge regularization")
plt.savefig("figures/x_hat_prox_ridge.png")


# compute quadratic error between original signal and x_hat_prox_ridge
quad_error = quadratic_error(x=signal, x_hat=x_hat_prox_ridge)
print("proximal gradient descend algo: \n "
      + "quadratique error x_bar vs x_hat_prox_ridge \n : ", quad_error)


# find the best lambda minimizing the error
lambdas = np.linspace(0.0001, 100, 20)
quadratic_errors_ridge = find_best_lambda(
    x=signal, y=y, K=K, lambdas=lambdas, gradient=gradient_f,
    ips=0.01, algo="descend_prox", regu="L2",
    output_dir="figures/quadratic_error_prox_ridge.png")

min_error = np.min(quadratic_errors_ridge)
best_lamb = lambdas[np.argmin(quadratic_errors_ridge)]

print(f"proximal grdt descend ridge case: min error ={min_error} for best lambda = {best_lamb}")


# B. Lasso regularization

# a. Compute x_hat using the Forward Backward Primal Dual algo
x_hat_lasso, objective_history_lasso = FB_primal_dual(
    y, K, lamb=0.1, tau=0.1, sigma=0.1, ips=0.01, regu="L1")


# plot objective_history_lasso
fig, ax = plt.figure(), plt.axes()
plt.scatter(np.arange(len(objective_history_lasso)),
            objective_history_lasso)
plt.title("objective fct value in each iteration of  \n"
          + "the forward backward primal dual descend algo \n"
          + "with lasso regularization")
plt.savefig("figures/objective_history_lasso.png")


# display x_hat_lasso
fig, ax = plt.figure(), plt.axes()
plt.plot(T, x_hat_lasso)
ax.set_yscale('log')
plt.title("estimated signal x_hat using \n"
          + "forward backward primal dual algo with ridge regularization")
plt.savefig("figures/x_hat_lasso.png")


# compute quadratic error between orginal and estimated signal
quad_error = quadratic_error(x=signal, x_hat=x_hat_lasso)
print("Forward Backward primal dual algo: \n "
      + "quadratique error x_bar vs x_hat_grdt_dscd_ridge \n : ", quad_error)


# find best lambda for optimization pb with lasso regularization
lambdas = np.linspace(0.0001, 100, 20)
quadratic_errors_lasso = find_best_lambda(
    x=signal, y=y, K=K, lambdas=lambdas, gradient=gradient_f,
    ips=0.01, algo="FB_primal_dual", regu="L1",
    output_dir="figures/quadratic_error_lasso.png")

min_error = np.min(quadratic_errors_lasso)
best_lamb = lambdas[np.argmin(quadratic_errors_lasso)]

print(f"FB primal dual with lasso case: min error ={min_error} for best lambda = {best_lamb}")

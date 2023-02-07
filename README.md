# Scope
Using different optimization algorithms including the gradient descend, the proximal gradient descend and the forward backward primal dual algorithms in order to estimate the optimal solution of the following optimization problem:
![Fig.3](./include/optim_pb.png "optimization problem")

# Context
DOSY method (Diffusion Order SpectroscopY) consists of a series of measurements acquired for magnetic pulses of increasing intensities. The degree of attenuation is then proportional to the diffusion coefficient of the molecule which is related to the size and shape of the latter. The data is then analyzed in order to separate the different species of the mixture according to their diffusion coefficient.

# Virtual environment
To run this project, please start by activate the built-in virtual environment. Once in the `spectroscopyrmn` directory, you can use the following command: source bin/activate. No other requirements are needed to run the `.py` files.


# Project
1. Input signal (x_bar) as function of a sampling grid T (x(T)) (see `functions.py`):
![Fig.1](./figures/original_signal.png "Original Signal")
\
2. y signal (y(t)) where ```y = K * x + w```
![Fig.2](./figures/y(t).png "y signal")
\
3. Finding an estimate (x_hat) of the original signal (x_bar) by solving the following optimization problem:
![Fig.3](./include/optim_pb.png "optimization problem")
#### a) With ridge regularization:
   ![Fig.4](./include/regu_term.png "regularization term: ridge")
   1. Gradient Descend Algorithm
   With ridge regularization term, we started by using the classic gradient descent algorithm in order to find the solution to this problem. We chose to stop our algorithm once the square of the norm L2 of the gradient is lower than a chosen epsilon value (10e-1, 10e-4), i.e. when the estimate x_hat is near the optimal solution. The decrease of the objective function value in each iteration when running this algorithm can be seen on the following plot: 
   ![Fig.5](./figures/objective_history_grdt_dscd_ridge.png "Objective fct history with gradient descend algorithm")
   The resulting x_hat signal: 
   ![Fig.6](./figures/x_hat_grdt_dscd_ridge.png "Estimated x_hat signal using gradient descend algorithm (with ridge regularization)")
   When searching for the best hyperparameter lambda, the quadratic error shows that the algorithm is not a good choice for our optimization problem.
   ![Fig.7](./figures/quadratic_error_grdt_dscd_ridge.png "Quadratic error x_bar vs x_hat using gradient descend algorithm")
   The latter results show that the gradient descend is not working well due to the fact that our matrix is **ill-conditioned**, i.e. its condition number is very large.\

   2. Proximal Gradient Descend Algorithm
   A more convenient algorithm to use in this situation is the proximal gradient descend algorithm.  Using, the same stop criteria, the decrease of the objective function value in each iteration when running this algorithm can be seen on the following plot: 
   ![Fig.8](./figures/objective_history_prox_ridge.png "Objective fct history with proximal gradient descend algorithm")
   The resulting x_hat signal: 
   ![Fig.9](./figures/x_hat_grdt_dscd_ridge.png "Estimated x_hat signal using proximal gradient descend algorithm (with ridge regularization)")
   The quadratic error between the original (x_bar) and the estimate (x_hat) signals for the best lambda (=10.53) is equal to ~0.719.
   ![Fig.10](./figures/quadratic_error_prox_ridge.png "Quadratic error x_bar vs x_hat using proximal gradient descend algorithm")

#### b) With lasso regularization:
   1. Forward Backward Primal Dual Algorithm
   When using the lasso regularization, both the proximal gradient descend and the gradient descend algorithms are not convenient. Thus, we chose to use the Forward Backward Primal Dual Algorithm.
   The decrease of the objective function value in each iteration when running this algorithm is as following:
   ![Fig.8](./figures/objective_history_lasso.png "Objective fct history with the forward backward primal dual algorithm")
   The resulting x_hat signal: 
   ![Fig.9](./figures/x_hat_lasso.png "Estimated x_hat signal using forward backward primal dual algorithm (with lasso regularization)")
   The quadratic error between the original (x_bar) and the estimate (x_hat) signals for the best lambda (=10.53) is equal to ~0.738
   ![Fig.10](./figures/quadratic_error_lasso.png "Quadratic error x_bar vs x_hat using forward backward primal dual algorithm")



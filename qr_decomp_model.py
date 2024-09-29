import pymc as pm
from numpy.linalg import qr, inv, LinAlgError
from numpy import sqrt,square, shape, std, exp
import sys

def meta_reg(X, y, ysd):
    """
    Perform regression using QR decompostion parameterisation PyMC.
    
    Parameters:
    X : np.ndarray
        Design matrix (predictors only i.e. no intercept term as qr decomp will not run).
    y : np.ndarray
        Response vector.

    Returns:
    trace : pm.trace
        The trace of the model after sampling.
    model : pm.Model
        The PyMC model object.
    """
    # Get the len of the y variable
    N = len(y) 
    K = shape(X)[1]
    half_K = .5 * K
    s_Y = std(y) 
    eta = .5
    sqrt_Nm1 = sqrt(N - 1.0)

    Q,_,R_inverse = qr_decomp_scale(X, N)

    # 
    se2=square(ysd)

    # Define the pymc model for linear regression with qr decomp based heavily on 
    # https://mc-stan.org/docs/stan-users-guide/regression.html#QR-reparameterization.section
    # https://github.com/stan-dev/rstanarm/blob/master/src/stan_files/lm.stan
    with pm.Model() as model:
        # Priors
        # coefficients on Q_as
        alpha = pm.StudentT("alpha", nu=3, mu=0.7, sigma=2.5)
        log_omega = pm.Flat("log_omega")
        R2 = pm.Beta("Beta", half_K, eta)

         # Define or generated quantities
        u_raw = pm.Flat("u_raw", shape=K)
        
        # Normalize the vector to have unit length
        u_unit = u_raw / pm.math.sqrt(pm.math.sum(u_raw**2))
    
        # Add a deterministic node to keep track of the unit vector
        u = pm.Deterministic('unit_vector', u_unit)

        delta_y = pm.Deterministic("delta_y", s_Y * exp(log_omega))
        theta = pm.Deterministic("theta", u * sqrt(R2) * sqrt_Nm1 * delta_y)
        sigma = pm.Deterministic("sigma", delta_y * sqrt(1 - R2))
        mu_qr = pm.Deterministic("mu_qr",
                              alpha + pm.math.dot(Q, theta) )

        # Likelihood
        likelihood = pm.Normal("y", mu = mu_qr, sigma = sqrt(square(sigma)+ se2),  observed=y)

        # Return calculated valeus using pm.Deterministic
        beta = pm.Deterministic("beta", pm.math.dot(R_inverse, theta))
        # Calculate mu on orignal scale.
        mu = pm.Deterministic("mu", alpha + X @ beta )

        trace = pm.sample(nuts_sampler="nutpie")
        pm.compute_log_likelihood(trace)
        return trace, model

def qr_decomp_scale(X, N):
    try:
        # Calculate the thin qr decompositon and the inverse of upper-triangular matrix R (defualt fucntion behaviour).
        Q,R = qr(X)
        # Scale Q and R.
        Q = Q * sqrt(N - 1)
        R = R / sqrt(N - 1)
        # Calculate the inverse of the upper triangular scaled R matrix from qr_decompisition. 
        R_inverse = inv(R)
        return Q,R,R_inverse
    
    # Handle error when X is not a matrix.
    except LinAlgError:
        sys.exit("Input to qr_decomp_scale is not a matrix") 
    except ValueError:
        sys.exit()

# Define main fucntion for testing of model.
def main():

    # Simualtion tes tof the model based on 
    # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM_linear.html 
    import numpy as np
    import arviz as az

    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    size = 200
    true_intercept = 1
    true_slope = 2
    true_slope2 = 3

    mean = 0   # Mean of the distribution
    std_dev = 1  # Standard deviation of the distribution
    array1 = np.random.normal(mean, std_dev, size)
    array2 = np.random.normal(mean, std_dev, size)
    # Stack the arrays column-wise
    X = np.column_stack((array1, array2))

    b = np.array([true_slope, true_slope2])

    # Compute y = a + X @ b (where @ denotes matrix multiplication)
    true_regression_line = true_intercept + X @ b
    ysd = rng.normal(scale=0.5, size=size)
    y = true_regression_line+ysd
   
    trace, model = meta_reg(X, y,ysd)
    print(az.summary(trace, var_names=["alpha", "beta"]))

if __name__ == "__main__":
    main()
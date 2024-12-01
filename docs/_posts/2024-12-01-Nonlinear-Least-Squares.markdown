---
layout: post
title:  "Nonlinear Least Squares"
date:   2024-12-01 10:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Nonlinear Least Squares

We previously saw that [Ordinary Least Squares]({% post_url 2024-09-17-Least-Squares %}) fits a linear model to data. Now we will consider the case where the phenomena being modeled is non-linear in it's parameters, meaning a linear relationship is insufficient to model the process. Suppose we would like to fit a polynomial, and even know the degree of the polynomial that should model our data. Then we can setup a system of Normal Equations by first considering the residuals between the observed phenomena output and the predicted output:

$$ 
r_i = y_i - f(x_i, \theta) \quad i \in \{i, \cdots, m \} 
$$

where $\theta$ are the polynomial coefficients. Using the sum of squares to evaluate the model fit,

$$
E = \sum_{i=1}^m r_i^2
$$

The model parameters can be estimated by taking the partial derivative of the error function with respect to each parameter and setting it to zero:

$$
\frac{\partial E}{ \partial \theta_j} = 2 \sum_i r_i \frac{\partial r_i}{ \partial \theta_j}
= 0
$$

for each of the `n` model paremeters. Call $\theta^k$ the current estimate of the model parameters, then we can estimate the model parameters by taking the Taylor polynomial expansion about this current estimate:

$$
f(x_i, \theta) \approx f(x_i, \theta^k) + \sum_j \frac{\partial f(x_i, \theta^k)}{\partial \theta_j} (\theta_j - \theta_j^k) 
= f(x_i, \theta^k) + \sum_j J_{ij} \Delta \theta_j
$$

where $J_{ij}$ is the Jacobian matrix entry corresponding to the partial derivative of the $i$th residual with respect to the $j$th model parameter. Defining the error between the observations and the current estimate as $\Delta y_i = y_i - f(x_i,\theta^k)$ then we can see that the expression for the $i$th residual is 

$$
r_i = y_i - f(x_i, \theta) = (y_i - f(x_i, \theta^k)) + (f(x_i, \theta^k) - f(x_i, \theta))
\approx \Delta y_i - \sum_j J_{ij} \Delta \theta_j
$$

And so plugging this into the expression for the partial derivative of the error function $E$ and setting it equal to zero:

$$
2 \sum_i J_{ij} (\Delta y_i - \sum_k J_{ik} \Delta \theta_k) = 0
$$

This yields the system of Normal equations:

$$
(J^T J) \Delta \theta = J^T \Delta y
$$

Solving this equation for $\Delta \theta$, the update expression can be used to estimate the model parameters:

$$ 
\theta^{k+1} = \theta^k + \Delta \theta 
$$

Coding this in Python as follows
{% highlight Python %}
def run_model_fitting(sample_size, beta, beta_true):
    converged = False
    iters = 0
    while not converged:
        x = 2 * np.random.rand(sample_size)
        y = np.array([special_function(xi) for xi in x])
        y_noisy = np.array([yi + np.random.normal(scale=0.02) for yi in y])
        f_hat = np.array([approximate_function(beta, xi) for xi in x])
        residuals = y_noisy - f_hat
        J = np.array([Jacobian(xi) for xi in x])
        # Solve JtJ * \Delta \theta = Jt * residuals
        # This is the Normal Equation
        JtJ_inv = np.linalg.inv(np.matmul(J.transpose(), J)) 
        delta_beta = np.matmul(JtJ_inv, np.matmul(J.transpose(), residuals))
        beta_error = np.linalg.norm(beta - beta_true)
        print(f"Beta: {beta} beta error: {beta_error} ... iteration {iters}")
        beta += delta_beta
        iters += 1
        if beta_error < 0.05:
            converged = True
    return {'sample_size': sample_size, 'iters': iters, 'beta': [beta]}
{% endhighlight %}

Where `special_function()` returns the ground truth model evaluation, `approximate_function()` evaluates the polynomial using the estimated parameters, and `Jacobian()` returns the model Jacobian. This plot shows the result of model fitting for various sample data sizes (number of observations) after convergence.
![NNLS](/images/Polynomial_fitting.png)

How does the size of the sample population (number of observations) affect convergence of this LS update scheme? Running 100 trials of parameter estimation where we vary the sample observation size, we can see that as the number of observations increases the number of iterations needed for convergence decreases:
![NNLS](/images/Average_iterations_vs_sample_size.png)
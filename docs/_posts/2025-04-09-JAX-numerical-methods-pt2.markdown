---
layout: post
title:  "Numerical Methods Review (Part II)"
date:   2025-05-09 08:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Numerical Methods Review (Part II)

In [Part I]({% post_url 2025-04-06-JAX-numerical-methods %}) we covered Euler and Newton's Methods.
This post will continue with the following topics:
* Gauss-Newton Method
* Levenberg-Marquardt Algorithm

# Gauss-Newton Method
Optimization method that iteratively minimizes some function using first-order information.
The main idea is to use the truncated Taylor polynomial of a function to construct an update scheme.
More concretly, suppose we would like to minimize some function $f: \mathbb{R}^n \rightarrow \mathbb{R}$.
Consider the Taylor expansion of this function around some input $x_a$:

$$
f(x) = f(x_a) + f'(x_a) * (x-x_a) + \frac{1}{2} f''(x_a) * (x-x_a)^2 + ...
$$

If we ignore anything higher than the first order derivative then we have 

$$
f(x) \approx f(x_a) + f'(x_a) * (x-x_a) 
$$

So our typical optimization problem can be approximated as

$$
\arg \min_{p} \frac{1}{2} ||f(x_k) + f'(x_k) * p||^2
$$

where $x_k$ is the current estimate of $x^*$, our desired solution.

if the matrix $f'(x_k)$ is full rank then $f'(x_k)^T f(x_k)$ is invertible, and one could solve for $p$
$p = - [f'(x_k)^T f(x_k)]^{-1} f'(x_k) f(x_k)$. 
Recall that the inverted matrix is the psuedo-inverse of $f'(x_k)$ and this update step is the least-squares solution. The solution at each iteration is solved for as $x_{k+1} = x_k + p_k$. Alternatively, one can use line search to further control the update step: $x_{k+1} = x_k + t_k p_k$. The benefit of Gauss-Newton is that we don't need second-order information.
In comparison, Newton's method does require second derivatives, and shows fast convergence around local minimums. This is a trade-off to save on computing/storing the Hessian matrix.

# Levenberg-Marquardt Algorithm
Levenberg-Marquardt (LV) is an extension of Gauss-Newton which adds a term to the cost function that limits the step size according to how much the cost function changes. If the cost function does not decrease sufficiently during some iteration, then the step size should be small so as to not move the optimizer in a suboptimal direction. LV provides robustness in the face of local minima that can trap Gauss-Newton. This is effectively a regularization method that enforces the estimates to stay relatively close to the current linearization point, since the local approximation is accurate in this area.

$$
\arg \min_{p} \frac{1}{2} ||f(x_k) + f'(x_k) * p||^2 + \lambda D ||p||^2
$$

The lambda parameter changes throughout the optimization, and controls the step size. If the step size reduces the norm of the 
loss function then lambda decreases. Otherwise, the update is rejected and lambda is inflated, forcing the next estimate to be closer to
the current position.

![Euler](/images/numerical_methods/LM_func_1_trajectory.png) 
![EulerZoom](/images/numerical_methods/LM_func_1_residuals.png)

*<medium> Figure: Levenberg-Marquardt method converging for function $F(x,y) = [cos(x) + sin(y); x*y]$ starting from point $[-60,60]$. 54 iterations are required until convergence to $1\mathrm{e}{-8}$ error tolerance. </medium>*

## Curve-Fitting with Nonlinear Least Squares

Suppose we wish to model some phenomena, for example the water levels at Ocean Beach during high and low tides.
We could start by collecting some (noisy) measurements of the water levels over some period of time. 
Next, we could guess some family of functions which seems to roughly model the phenomena, 
for example some combination of sinusoidal functions.
Finally, we could use least squares to estimate the parameters that best model the observations.
This process is called curve-fitting, and we can use nonlinear optimization methods to estimate our model parameters.
More concretely, we have some measurement observations $y_t$ and some function we're trying to fit to the data 
$\hat{y}(t;a)$, where $a$ are the function parameters we would like to estimate. The loss function we'll use for this 
optimization is $e(t) = \sum_{t} y_t - \hat{y}(t;a)$.


In this experiment we use Levenberg-Marquardt and Gradient Descent to fit a curve to the function

$$
f(t) = a_1 \exp(-\frac{t}{a_2}) + a_3 t \exp(-\frac{t}{a_4})
$$

We start by corrupting the truth function values with noise drawn from the distribution $\mathcal{N(0,0.2)}$.
This simulates measurement error inherent to observing any real world phenomena. We use these noisy estimates to fit our model.
These [notes][curve-fitting] were very informative for this implementation.

This Python code uses Jax, an library for numerical computation that brings the following major benefits:
 - Automatic differentiation
 - JIT compilation 
 - NumPy-like interface

First, generate some data that has been corrupted as described above: 
{% highlight Python %}
class DataGenerator:
    def __init__(self, coefficients : jnp.array):
        self.times : jnp.array
        self.values : jnp.array
        self.measurements : jnp.array
        self.coefficients = coefficients
        self.noise_mu = 0.0
        self.noise_sigma = 0.2
 
    def generate(self):
        self.times, self.values = self.generate_data()

    def collect_observations(self):
        key = random.PRNGKey(32)
        _, subkey = random.split(key)
        noise = random.normal(subkey, shape=self.values.shape)
        scaled_noise = self.noise_mu + self.noise_sigma * noise
        self.measurements = self.values + scaled_noise
        return self.measurements
    
    def generate_data(self):
        t = jnp.linspace(0, 100, 500)
        return (t, model(self.coefficients, t))
{% endhighlight %}

The function we're sampling from and trying to model:
{% highlight Python %}
def model(a: jnp.array, t : jnp.array):
    '''
    Sample data generated from function:
        F(t) = a1 exp(-t/a2) + a3 t exp(-t/a4)
    '''
    return a[0] * jnp.exp(-t/a[1]) + a[2] * t * jnp.exp(-t / a[3])
{% endhighlight %}

An abstract optimization class. Under the hood it can use any optimization method.
{% highlight Python %}
class Optimizer:
    def __init__(self, use_lm_opt : bool = True):
        self.max_iterations = 1000
        self.convergence_threshold = 1e-8
        if use_lm_opt:
            self.opt_method = LevenbergMarquardt()
        else:
            self.opt_method = GradientDescent()

    def optimize(self, loss, a_init : jnp.array, t : jnp.array, plot_opt_results):
        residuals = np.zeros((self.max_iterations,1))
        coeffs = np.zeros((self.max_iterations, a_init.shape[0]))
        ak = a_init
        it = 0
        residual_delta = float('inf')
        while ((jnp.linalg.norm(loss(ak,t)) > self.convergence_threshold and
               residual_delta > self.convergence_threshold) and 
               it < self.max_iterations):
            residuals[it] = jnp.linalg.norm(loss(ak,t))
            coeffs[it,:] = ak
            ak = self.opt_method.estimate(loss, {'a':ak, 'x':t})
            if it > 0:
                residual_delta = jnp.linalg.norm(residuals[it]-residuals[it-1])
            it += 1

        if plot_opt_results:
            self.plot_results(it, loss, residuals, coeffs, t)

        return (coeffs[:it,:], it)
{% endhighlight %}

Implementation of Gradient Descent and Levenberg-Marquardt algorithms.
{% highlight Python %}
class GradientDescent:
    def __init__(self):
        self.alpha = 1e-3

    def compute_step(self, f, a, x):
        Jt = jnp.transpose(jax.jacfwd(f, argnums=0)(a,x))
        return -self.alpha * jnp.matmul(Jt, f(a,x))
    
    def estimate(self, f, params):
        a = params['a']
        x = params['x']
        return a + self.compute_step(f, a, x)

class LevenbergMarquardt:
    def __init__(self):
        self.lam = 1.
        self.lambda_history = []
        self.lam_inflation_factor = 2
        self.lam_deflation_factor = 3

    def compute_step(self, f, a, x):
        # g(x) = ||f(x)||^2
        # grad_g(x) = f_prime(x)^T * f(x)
        # J = f_prime
        Jt = jnp.transpose(jax.jacfwd(f, argnums=0)(a,x))
        M = jnp.matmul(Jt, jnp.transpose(Jt)) + self.lam * jnp.eye(Jt.shape[0])
        grad_g = jnp.matmul(Jt, f(a,x))
        return -1./2 * jnp.matmul(jnp.linalg.inv(M), grad_g)

    def estimate(self, f, params):
        a = params['a']
        x = params['x']
        v = self.compute_step(f, a, x)
        f_ak = f(a + v, x)
        f_a = f(a, x)
        self.lambda_history.append(self.lam)
        if jnp.linalg.norm(f_ak) <= jnp.linalg.norm(f_a):
            ak = a + v
            self.lam /= self.lam_deflation_factor
        else:
            ak = a
            self.lam *= self.lam_inflation_factor
        return ak
{% endhighlight %}

The initial starting coefficents of the model are generated randomly. We then minimize our loss function, using both Levenberg-Marquardt and Gradient Descent independently. The following plots show results across three trials, each starting from a different initial condition.
The conclusion from this simulation is that LM requires less steps than GD to converge to a solution, and both methods are sensitive to the initial starting condition.

### Trial 1
![LM_residuals_T1](/images/numerical_methods/LM_residual_errors_T1.png)
![LM_manifold_T1](/images/numerical_methods/LM_curve_and_hist_T1.png) 
![LM_hist_T1](/images/numerical_methods/LM_trajectory_manifold_T1.png) 
*<medium> <b>Figure</b>: Method: Levenberg-Marquardt <br>
Initial guess: $[0.2859322,0.6534275,0.32196856,0.9826077]$ <br> 
Final estimate: $[3.92110848,2.73187017,2.0193162,9.94372177]$ <br>
Convergence in 19 iterations </medium>*

![DG_residuals_T1](/images/numerical_methods/GD_residual_errors_T1.png){:width="420px"} 
![DG_manifold_T1](/images/numerical_methods/GD_curve_and_hist_T1.png)
![DG_hist_T1](/images/numerical_methods/GD_trajectory_manifold_T1.png) 
*<medium> <b>Figure</b>: Method: Gradient Descent <br>
Initial guess: $[0.2859322, 0.6534275, 0.32196856, 0.9826077]$ <br> 
Final estimate: $[3.91499424, 2.72453666, 2.02070355, 9.94066334]$ <br>
Convergence in 923 iterations </medium>*

### Trial 2
![LM_residuals_T2](/images/numerical_methods/LM_residual_errors_T2.png) 
![LM_manifold_T2](/images/numerical_methods/LM_curve_and_hist_T2.png) 
![LM_hist_T2](/images/numerical_methods/LM_trajectory_manifold_T2.png) 
*<medium> <b>Figure</b>: Method: Levenberg-Marquardt <br> 
Intial guess: $[0.1688324, 0.64583564, 0.6233872, 1.0743146]$ <br>
Final estimate: $[3.92113495, 2.73185682, 2.01933193, 9.94373894]$ <br>
Convergence in 19 iterations</medium>*

![DG_residuals_T2](/images/numerical_methods/GD_residual_errors_T2.png){:width="420px"}
![DG_manifold_T2](/images/numerical_methods/GD_curve_and_hist_T2.png)  
![DG_hist_T2](/images/numerical_methods/GD_trajectory_manifold_T2.png) 
*<medium> <b>Figure</b>: Method: Gradient Descent <br>
Initial guess: $[0.1688324, 0.64583564, 0.6233872, 1.0743146]$ <br>
Final estimate: $[3.91576958, 2.72666121, 2.02038097, 9.94137669]$ <br>
Convergence in 965 iterations </medium>*

### Trial 3
![LM_residuals_T3](/images/numerical_methods/LM_residual_errors_T3.png) 
![LM_manifold_T3](/images/numerical_methods/LM_curve_and_hist_T3.png) 
![LM_hist_T3](/images/numerical_methods/LM_trajectory_manifold_T3.png) 
*<medium> <b>Figure</b>: Catastrophic failure. Method: Levenberg-Marquardt <br>
Initial guess: $[1.6456262, 1.7493442, 0.1221709, 0.30687004]$ <br> 
Final estimate: $[4.27152967, 29.75058365, -0.24114212, 0.37449351]$ <br>
Convergence in 4 iterations </medium>*


## Conclusions
LM shows impressive results, requiring far fewer iterations to reach an optimal solution compared to GD.
I acknowledge that GD would likely perform better with improved selection of the learning rate parameter.

Next steps include:
 * Add an adaptive learning rate to GD to improve convergence robustness.
 * Apply LM optimization to fit a model to NOAA tidal data, or other scientific phenomena.

[curve-fitting]: https://people.duke.edu/~hpgavin/lm.pdf 

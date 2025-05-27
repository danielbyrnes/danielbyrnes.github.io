---
layout: post
title:  "Numerical Methods Review (Part II)"
date:   2025-06-09 08:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Numerical Methods Review (Part II)

In [Part I]({% post_url 2025-04-06-JAX-numerical-methods %}) we covered Euler and Newton's Methods.
This post will continue with the following topics:
* Gauss-Newton Method
* Levenberg-Marquardt Algorithm
* Cholesky Factorization 
    - Gram-Schmidt

# Gauss-Newton Method
Optimization method that iteratively minimizes some function using first-order information.
The main idea is to use the truncated Taylor polynomial of a function to construct an update scheme.
More concretly, suppose we would like to minimize some function $f: \matbb{R}^n -> \mathbb{R}$.
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
\argmin_{p} \frac{1}{2} ||f(x_k) + f'(x_k) * p||^2
$$

where $x_k$ is the current estimate of $x^*$, our desired solution.

if the matrix $f'(x_k)$ is full rank then $f'(x_k)^T f(x_k)$ is invertible, and one could solve for $p$
$p = - [f'(x_k)^T f(x_k)]^{-1} f'(x_k) f(x_k)$. Recall that the inverted matrix is the psuedo-inverse of $f'(x_k)$, and this update step is the least-squares solution. The solution at each iteration is solved for as $x_{k+1} = x_k + p_k$. Alternatively, one can use line search to further control the update step: $x_{k+1} = x_k + t_k p_k$. The benefit of Gauss-Newton is that we don't need second-order information.
In comparison, Newton's method does require second derivatives, and shows fast convergence around local minimums. This is a trade-off to save on computing/storing the Hessian matrix.

# Levenberg-Marquardt Algorithm
Levenberg-Marquardt (LV) is an extension of Gauss-Newton which adds a term to the cost function that limits the step size according to how much the cost function changes. If the cost function does not decrease sufficiently during some iteration then the step size should be small so as to not move the optimizer in a suboptimal direction. LV provides robustness in the face of local minima that can trap Gauss-Newton. This is effectively a regularization method that enforces the estimates to stay relatively close to the current linearization point, since the local approximation is accurate in this area.

\argmin_{p} \frac{1}{2} ||f(x_k) + f'(x_k) * p||^ + \lambda D ||p||^22

The lambda parameter changes throughout the optimization, and controls the step size.

# Cholesky Factorization 
This matrix factorization is suitable for hermitian positive definite matrices. If we limit our consideration to matrices in real space then we can simply say symmetric positive definite matrices.

#### Gram-Schmidt


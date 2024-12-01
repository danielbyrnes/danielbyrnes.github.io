---
layout: post
title:  "Line Fitting with Least Squares"
date:   2024-09-17 20:07:43 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

This post illustrates using the method of least squares ([LS][ls-gtech]) to fit a line to some noisy data.
Given a linear system $Ax = b$, the method of least squares generates a solution that minimizes the sum of square differences of $b - Ax$, also called the residuals.

## Ordinary Least Squares

We will start with Ordinary Least Squares (OLS), where we hope to fit a linear model to some data and make some assumptions about sample independence. Consider modeling a linear process 

$$
y_i = m \cdot x_i + b + \epsilon
$$

where $\epsilon$ is some normally distributed noise that captures measurement and model error.
Ignoring the error term for now, we can stack the observations of our dependent and independent variables as follows

$$
\begin{bmatrix}
y_1 \\
\vdots \\
y_n
\end{bmatrix}
= 
\begin{bmatrix}
1 & x_1 \\
\vdots & \vdots \\
1 & x_n
\end{bmatrix}
\begin{bmatrix}
b \\
m
\end{bmatrix}
$$

Denoting the data matrix (containing the $x_i$'s) as $X$ and the observations as vector $y$, this expression can be re-written as

$$ 
y = X
\begin{bmatrix}
b \\
m
\end{bmatrix}
$$

Multiplying both sides by $X^T$ we get the Normal Equations:

$$ 
X^T y = X^T X
\begin{bmatrix}
b \\
m
\end{bmatrix}
$$

$$ 
\begin{bmatrix}
b \\
m
\end{bmatrix}
 = (X^T X)^{-1} X^T y
$$

Plugging in the definition of $X$ we get the following 

$$
X^T X = 
\begin{bmatrix}
n & n \bar{x} \\
n \bar{x} & \sum x_i^2
\end{bmatrix}
$$

and 

$$
X^T y = 
\begin{bmatrix}
\sum y_i \\
\sum x_i y_i
\end{bmatrix}
$$

Here is some Python code to illustrate LS. First, let's import a few required modules:
{% highlight Python %}
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
{% endhighlight %}

Generate some synthetic input data with added noise, then run LS and plot the results.
{% highlight Python %}
def least_squares_main():
    coefficients = 10 * np.random.rand(2)
    m, b = coefficients
    num_points = 200
    data = 5 * np.random.rand(num_points, 2)
    true_data = np.copy(data)
    for i in range(num_points):
        true_data[i,1] = m * data[i,0] + b 
        data[i,1] = true_data[i,1] + np.random.normal(0, 0.25) # ~N(mu=0,sigma=0.25)

    m_est, b_est = solve_least_squares(data)   
    
    ls_estimate_data = np.copy(data)
    for i in range(num_points):
        ls_estimate_data[i,1] = m_est * ls_estimate_data[i,0] + b_est  
    plot_data(data, ls_estimate_data, true_data)

    print(f"m: {m}, b: {b}")
    print(f"m error: {m_est - m}, b error: {b_est - b}")

if __name__ == '__main__':
    least_squares_main()
{% endhighlight %}

Estimate the linear model parameters using SVD:
{% highlight Python %}
def solve_least_squares(data):
    M = np.zeros((2,2))
    b = np.zeros((2,1))
   
    for i in range(data.shape[0]):
        M[0,0] += data[i,0] * data[i,0]
        M[0,1] += data[i,0]
        b[0] += data[i,0] * data[i,1]
        b[1] += data[i,1]
    M[1,0] = M[0,1]
    M[1,1] = data.shape[0]

    # Solve x = inv(M) * b
    #x = np.linalg.inv(M).dot(b)
    U, s, Vh = linalg.svd(M)
    x = U.T.dot(b)
    x = np.diag(1/s).dot(x)
    x = np.conjugate(Vh).T.dot(x)
    assert x.shape == (2,1)
    x = np.ravel(x)
    return x[0], x[1]
{% endhighlight %}

The results will be plotted as such:
{% highlight Python %}
def plot_data(data, ls_estimate_data, true_data):
    plt.plot(true_data[:,0], true_data[:,1], color='k')
    plt.scatter(data[:,0], data[:,1], color='red')
    plt.plot(ls_estimate_data[:,0], ls_estimate_data[:,1], color='green')
    plt.show()
{% endhighlight %}

![LS](/images/LS_simulation.png)

Estimated paramters of the linear model: `m=5.25, b=1.97`. \\
Error of the model parameters: `m error=-0.0023, b error=0.0166`.

It is worth noting that the OLS estimate is `unbiased` and `consistent` if the errors have finite variance and are uncorrelated with the independent variables (the $x_i$'s). An estimator is said to be `unbiased` if the expected value of its estimates are equal to the true values of the parameters being estimated. In other words, the estimator is on average correct and is not systematically producing an over/under estimate. An estimator is `consistent` if the estimate converges to the true value as the size of the data (the observations) increases towards infinity. In other words, with more data your estimate *only* becomes *more* accurate. [<cite>[UTDallas Estimation][1]</cite>]

## Robustification
We previously saw that OLS can model linear phenomena in the presence of error, given certain assumptions about the distribution of the error terms $e_i$. Often we cannot solve $Ax=b$ exactly, and the best we can do is instead minimize the error

$$
\underset{x \in \mathbb{R}^n}{\text{min}} || Ax - b ||_2^2
$$

This problem can be solved using the SVD of $A$, as seen above. If the smallest singular value of $A$ is significantly smaller than the largest, then the condition number will be huge and the matrix is ill-conditioned. This means that small perturbations to $b$ can yield large changes in the estimated solution. For this reason sometimes `Regularized Least Squares` are used instead, where the term $\frac{\lambda}{2} \lVert x \rVert_2^2$ is added to the cost function during the optimization problem. This has the effect of introducing sparsity into the estimated solution such that components corresponding to small singular values are nearly eliminated, while components corresponding to singular values relatively large compared to the regularization term $\lambda$ are left intact. In other words, $x_i \rightarrow 0$ for $\sigma_i \ll \lambda$.

## Up Next

Next we will consider fitting a model to non-linear data in this post about [Nonlinear Least Squares]({% post_url 2024-12-01-Nonlinear-Least-Squares %}).

[ls-gtech]: https://textbooks.math.gatech.edu/ila/least-squares.html
[1]: https://personal.utdallas.edu/~scniu/OPRE-6301/documents/Estimation.pdf

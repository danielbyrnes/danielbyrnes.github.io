---
layout: post
title:  "Line Fitting with Least Squares"
date:   2024-09-17 20:07:43 -0500
categories: jekyll update
---

This post illustrates using the method of least squares ([LS][ls-gtech]) to fit a line to some noisy data.
Given a linear system `Ax = b`, the method of least squares generates a solution that minimizes the sum of square differences of `b - Ax`.

Here is some Python code to illustrate LS. First, let's import a few required modules:
{% highlight Python %}
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
{% endhighlight %}

The results will be plotted as such:
{% highlight Python %}
def plot_data(data, ls_estimate_data, true_data):
    plt.plot(true_data[:,0], true_data[:,1], color='k')
    plt.scatter(data[:,0], data[:,1], color='red')
    plt.plot(ls_estimate_data[:,0], ls_estimate_data[:,1], color='green')
    plt.show()
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
        data[i,1] = true_data[i,1] + random.uniform(-1,1)

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
def solve_least_squares(x):
    M = np.zeros((2,2))
    b = np.zeros((2,1))
   
    for i in range(x.shape[0]):
        M[0,0] += x[i,0] * x[i,0]
        M[0,1] += x[i,0]
        b[0] += x[i,0] * x[i,1]
        b[1] += x[i,1]
    M[1,0] = M[0,1]
    M[1,1] = x.shape[0]

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

![LS](/images/LS_simulation.png)

Estimated paramters of the linear model: `m=3.58, b= 9.89`.  
Error of the model parameters: `m error=0.04, b error=-0.13`.

[ls-gtech]: https://textbooks.math.gatech.edu/ila/least-squares.html

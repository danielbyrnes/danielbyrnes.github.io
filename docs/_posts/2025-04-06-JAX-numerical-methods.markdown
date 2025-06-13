---
layout: post
title:  "Numerical Methods Review (Part I)"
date:   2025-04-07 08:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Outline 
In this post we will review some common numerical methods topics, and provide some Python code snippets. As we've seen previosuly, the [JAX]({% post_url 2025-03-19-JAX-intro %}) framework provides nice autodiff capabilities that we'll take advantage of.
The outline of topics is as follows:
* Euler Method
* Newton's Method
* Gauss-Newton Method
* Levenberg-Marquardt Algorithm

This post will cover the first two topics.

# Euler Method
This is an integration technique. Given a differential equation and initial value condition, this method estimates a function that satisfies both these criteria. More concretely, consider an unknown function $y(t)$ that varies with time. Suppose you only had a differential equation $\dot y(t)$ and some initial value $y(t_0) = y_0$. If there isn't an analytical expression to solve this differential equation, or solving it is cumbersome, then you can use Euler's method to compute an approximate solution.

The Euler method is an iterative method that consists of creating tangent lines that approximate the shape of a function ($y(t)$ in our example). We can write an expression for the line tangent to the function at the initial value. In this 1-D example this looks like 

$$
y(t) - y(t_0) = y'(t_0) (t - t_0)
$$

The distance of $t$ to $t_0$ controls the degree to which we approximate $y(t)$ by a line in the area around $t_0$; choosing $t$ very close to $t_0$ should incur a small approximation error assuming local linearity around this point. If we fix the step size to some constant $h$ (such that $t_{k+1} = t_k + h$)  then we can write the following update formula:

$$y_{t_{k + 1}} = y_{t_k} + y'(t_k) * (t_k + h - t_k) = y_{t_k} + y'(t_k) h$$

where $y_{t_k}$ is our approximation of $y(t_k)$.

{% highlight Python %}
def EulerMethod(func, x0, h, n):
    '''
    func: Function to differentiate.
    x0: Initial condition.
    h: Step size.
    n: Domain boundary. 
    '''
    f_dot = jax.grad(func)
    num_intervals = int(n / h)
    ys = np.zeros((num_intervals,1))
    ys[0] = func(x0) # Initial value
    for i in range(1,num_intervals):
        ys[i] = ys[i-1] + h * float(f_dot(x0 + i * h))
    return ys
{% endhighlight %}

![Euler](/images/numerical_methods/eulers_method.png) 
![EulerZoom](/images/numerical_methods/eulers_method_zoom.png)

*<medium> Figure: (Top) Euler's method using various step sizes to approximate function $f(x) = \sin(x) \cos(18x) + x^2$. (Bottom) Zoomed in; using $h=0.005$ approximates the function pretty well. </medium>*

To learn about more numerical integration methods take a look at [Runge-Kutta][runge-kutta].

# Newton's Method
This is a method to find the roots of a function $f(x)$, or in other words the points at which $f(x)=0$. 
This is ubiquitous problem within science and engineering, where for example you may need to solve some system like $Ax=0$.

Similar to the previous method, we can examine the tangent line at a given point along some function $f(x)$. If we imagine extending this line until it intersects the x axis, we can write the expression of the slope of this tangent:

$$
slope = \dfrac{f(x_0) - 0}{x_0 - x_1}
$$ 

where we exploit the detail of the tangent line at $x_1$.
Substituting the gradient of the function at point $x_0$ and rearranging we get the following

$$
x_1 = x_0 - \dfrac{f(x_0)}{f'(x_0)}
$$ 

This can be generalized to the update scheme $x_{n+1} = x_n - \dfrac{f(x_n)}{f'(x_n)}$.

The slope will effectively guide us towards some fixed point where the update scheme will converge.

{% highlight Python %}
def NewtonsMethod_FindZeros(func, x0, max_iters = 100, eps = 1e-6):
    '''
    func: Function to differentiate.
    x0: Initial condition.
    max_iters: Maximum number of iterations.
    eps: Convergence stopping condition. 
    '''
    f_dot = jax.grad(func)
    xs = np.zeros((max_iters,))
    xs[0] = x0
    i = 1
    while i < max_iters:
        xs[i] = xs[i-1] - (1./f_dot(xs[i-1])) * func(xs[i-1])
        if abs(xs[i] - xs[i-1]) < eps:
            break
        i += 1
    return xs[0:i-1]
{% endhighlight %}

![NewtonZeroF](/images/numerical_methods/newtons_method_F.png) 
![NewtonZeroF2](/images/numerical_methods/newtons_method_F2.png)

*<medium> Figure: (Top) Running Newton's method on function $f(x) = \sin(x) \cos(18x) + x^2$ starting from $x=0.7$ converges to $x=0.092415$ in 5 iterations. (Bottom) Starting from $x=0.6$ for function $f(x) = 20 \cos(0.1 x + 4.5) \sin(x + 1.4)$ converges to $x=1.74159$ in 5 iterations. </medium>*

#### Application to Optimization
Recall from calculus that if we wanted to find the maximum or minimum of a function then we should set its derivative to zero and solve, $f'(x) = 0$. We can apply Newton's method to find the zeros of the derivative of a function, essentially solving an optimization problem. Of course, we still need second-order information to know if we found a minimum, maximum, or saddle point. In 1-D the update equation becomes 

$$
x_{k+1} = x_k - (f''(x_k))^{-1} f'(x_k)
$$

Another way to see this is to use the Taylor expansion of the function $f(x_k + h)$:

$$
f(x_k + h) \approx f(x_k) + f'(x_k) h + \dfrac{1}{2} f''(x_k) h^2
$$

Setting the *derivative* of the right hand side to zero and solving for $h$ we can see

$$
f'(x_k) + f''(x_k) h = 0 \implies h = - \dfrac{f'(x_k)}{f''(x_x)}
$$

And set $x_{k+1} = x_k + h$.

![NewtonMinF5](/images/numerical_methods/newtons_method_optimize_F5.png)

*<medium> Figure: Newton's method converging from $x=9.5$ to $x=-0.3685$ in 4 iterations for function $f(x) = e^{\tfrac{1}{3}x} + \tfrac{2}{5} x^2 - 7$.</medium>*

Generalizing this to higher dimensional space just requires using the gradient and Hessian in place of the first and second derivates.

![NewtonOpt3D](/images/numerical_methods/newtons_method_optimize_3d.png)

*<medium> Figure: Newton's method converging for (non-convex) function $f(x,y) = x^2 y^2$ starting from point $[-60,60]$. 54 iterations are required until convergence to $1\mathrm{e}{-8}$ error tolerance. </medium>*


[runge-kutta]: https://en.wikipedia.org/wiki/Runge–Kutta_methods

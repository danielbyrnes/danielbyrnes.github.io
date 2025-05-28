---
layout: post
title:  "Linear Programming with Simplex Method"
date:   2025-05-26 08:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Linear Programming
Linear programming is a class of constrained optimization problems where the objective (or cost) function is a linear function of some variables, and a series of constraints must be satisfied. The problem typically takes the form:  

$$
\begin{align*}
    \text{max} \quad
    c^T x& \\
    \text{s.t.} \quad
    Ax &\leq b \\
    x &\geq 0
\end{align*}
$$

where $A$ is an $n \times m$ integer matrix representing $n$ constraints and $m$ variables.

## Simplex Method
George Dantzip developed the Simplex method for solving LPs while he was working for the US Air Force.
The basic idea is to turn the LP constraints into equalities by introducing *slack variables*, which introduces a trivial solution to the problem (the $m$ objective function variables are zero). It turns out that the set of all possible solutions is a convex polytope (potentially unbounded), and the optimal solution (if it exists) is one of its vertices. 

The set of all possible solutions is called the *feasible region*, and a vertex of the polytope is called a *basic feasible solution*. The Simplex method is a systematic process for searching through the polytope vertices to find the optimal solution without having to visit every single vertex. Each iteration of the algorithm performs a *pivot* operation that selects a new vertex of the polytope that increases the objective function.

Note that the constraints in the LP are all "<=". This is the standard form. If we added ">=" constraints then we could multiply the constraint by $-1$ to flip the equality sign, but then our standard trivial solution (the zero vector) would not feasible (since 0 is not less than a negative number). 
So we need a different polytope vertex to kickstart the Simplex algorithm. In order to support ">=" constraints we need to add *artificial variables* to the tableau and run Simplex using an auxiliary objective function. This function is just the negative sum of the artificial variables: $-a_1 - \cdots -a_r$, where there are $r$ ">=" constraints in the LP. 

Minimizing this auxiliary objective function will ensure that we find a basic feasible solution where all the added artificial variables are 0. Once we have this basic feasible solution, we can replace the cost function in the tableau with the original LP cost and then run Simplex again. This is what wen refer to as the II-Phase Simplex method.

![LP1](/images/linear_programming_p1.png)
*<medium>(Fig) Feasible region where the objective function is to maximize $x1 + x2$. The optimum is found at the vertex $(2,3)$ and has value 5. </medium>*

## Primal vs. Dual Method
Every LP has a counterpart called the *dual form*. There are theoretical guarantees that if the optimum value of the primal LP is finite then the dual will have the same optimum. Using the same variables from the primal LP defined above, we can define the dual optimization problem: 

$$
\begin{align*}
    \text{min} \quad
    b^T y& \\
    \text{s.t.} \quad
    A^Ty &\geq c \\
    y &\geq 0
\end{align*}
$$

The dual problem can provide a different perspective of the same problem, and is sometimes more efficient to solve. We will utilize the dual formulation for solving minimization problems.

## Forming the Tableau
formulating the tableau is done as follows. As noted previously, for minimization problems we solve the dual instead of the primal problem
```c++
Eigen::MatrixXd SimplexMethod::FormTableau(const Eigen::MatrixXd& CC,
                            const Eigen::VectorXd& cl,
                            const Eigen::VectorXd& cost) {
    // Use dual simplex method for minimization problems
    if (optimize_dual_) {
        // Form tableau for dual problem
        Eigen::MatrixXd CC_t = CC.transpose();
        uint32_t num_constraints = CC_t.rows();
        uint32_t num_vars = CC_t.cols();
        nc_ = num_constraints;
        nv_ = num_vars;
        // General tableau form dual problem:
        //       CC^T  sl cost
        //       -cl   0  0
        uint32_t sv = nv_; // num variables
        Eigen::MatrixXd dual_tableau = Eigen::MatrixXd::Zero(nv_+1,nv_+nc_+1);
        dual_tableau.block(0,0,nv_,nc_) = -CC_t;
        dual_tableau.block(0,nc_,sv,sv) = Eigen::MatrixXd::Identity(sv, sv);
        dual_tableau.block(0,nc_+sv,nv_,1) = -cost;
        dual_tableau.block(nv_,0,1,nc_) = cl.transpose();
        return dual_tableau;
    }
    uint32_t num_constraints = CC.rows();
    uint32_t num_vars = CC.cols();
    assert(num_constraints == nc_);
    assert(num_vars == nv_);
    uint32_t sv = nc_; // num slack variables
    Eigen::MatrixXd tableau = Eigen::MatrixXd::Zero(nc_+1,nv_+nc_+1);
    // General tableau form (for maximization problem):
    //        CC  sl cl
    //      -cost 0  0
    tableau.block(0,0,nc_,nv_) = CC;
    tableau.block(0,nv_,sv,sv) = Eigen::MatrixXd::Identity(sv, sv);
    tableau.block(0,nv_+sv,nc_,1) = cl;
    tableau.block(nc_,0,1,nv_) = -cost.transpose();
    return tableau;
}
```

## Pivoting
Let's introduce some more terminology. A *basic* variable is one that appears as a standard basis vector in the tableau. These variables have non-zero values in the current feasible solution. The remaining $n$ variables are called *nonbasic* variables, and are set to zero in the current feasible solution.
Any $n+m$ feasible solution will have $n$ positive values. The goal of the pivoting operation is to replace one of our basic variables with one that is nonbasic such that the objective function increases. 
More concretely, the pivoting operation consists of selecting a column corresponding to a nonbasic variable to make basic, and a row corresponding to a nonbasic variable to exit the solution basis.
After the pivot operation is complete, the new basis should correspond to another extrema (or vertex) of the feasible region.

The pivoting code in our `SimplexMethod` class looks as follows.
```c++
while (true) {
        // Find the pivot column
        Eigen::Index pivot_col;
        // For maximization problems select the column with the
        // most negative entry in the objective row
        tableau.row(nc).head(nv).minCoeff(&pivot_col);
        if (tableau(nc, pivot_col) >= 0) {
            break;
        }

        // Find pivot row
        double ratio_limit = std::numeric_limits<double>::max();
        uint32_t pivot_row = 0;
        for (uint32_t row = 0; row < nc; ++row) {
            if (tableau(row, pivot_col) > 0) {
                double ratio = tableau(row, nv) / tableau(row, pivot_col);
                if (ratio >= 0 && ratio < ratio_limit) {
                    ratio_limit = ratio;
                    pivot_row = row;
                }
            }
        }
    ...
```
Finally, the pivot operation makes the selected column one of the basic variables:
```c++
// Perform pivot operation
double pivot_val = tableau(pivot_row, pivot_col);
tableau.row(pivot_row) /= pivot_val;
for (uint32_t r = 0; r < tableau.rows(); ++r) {
    if (r == pivot_row) continue;
    double scale_factor = tableau(r, pivot_col);
    tableau.row(r) -= scale_factor * tableau.row(pivot_row);
}

```
## Optimization Interface
The interface for setting up the LP should be intuititve, where we can add >= and <= constraints and specify whether the cost function should be maximized or minimized:
```c++
Optimizer optimizer;
optimizer.AddLTConstraint({1, 2}, 8);
optimizer.AddLTConstraint({3, 2}, 12);
optimizer.AddGTConstraint({1, 3}, 3);
optimizer.MaximizeCost({1, 1});
auto result = optimizer.Solve();
PrintSolution(3, result);
```

which yields the output
```
Maximize 1 * x_1 + 1 * x_2
 such that 
	1 * x_1 + 2 * x_2 <= 8
	3 * x_1 + 2 * x_2 <= 12
	-1 * x_1 + -3 * x_2 <= -3

############################################################
(2) Optimized solution: 2 3 0 0 8 ---> optimum @ 5
############################################################
```

Similarly, we can define a minimization problem:
```c++
Optimizer optimizer;
optimizer.AddGTConstraint({1, 1, 1}, 6);
optimizer.AddGTConstraint({0, 1, 2}, 8);
optimizer.AddGTConstraint({-1, 2, 2}, 4);
optimizer.MinimizeCost({2, 10, 8});
auto result = optimizer.Solve();
PrintSolution(7, result);
```
which yields the output
```
Minimize -2 * x_1 + -10 * x_2 + -8 * x_3
 such that 
	-1 * x_1 + -1 * x_2 + -1 * x_3 <= -6
	-0 * x_1 + -1 * x_2 + -2 * x_3 <= -8
	1 * x_1 + -2 * x_2 + -2 * x_3 <= -4

############################################################
(7) Optimized solution: 0 0 2 2 0 4 ---> optimum @ 36
############################################################
```

Here are some of our other test cases:
```
Maximize 3 * x_1 + 5 * x_2
 such that 
	1 * x_1 + 0 * x_2 <= 4
	0 * x_1 + 2 * x_2 <= 12
	3 * x_1 + 2 * x_2 <= 18
	-3 * x_1 + -2 * x_2 <= -2

############################################################
(1) Optimized solution:  2  6  2  0  0 16 ---> optimum @ 36
############################################################

Maximize 20 * x_1 + 30 * x_2
 such that 
	1 * x_1 + 1 * x_2 <= 7
	1 * x_1 + 2 * x_2 <= 12
	2 * x_1 + 1 * x_2 <= 12

############################################################
(3) Optimized solution: 2 5 0 0 3 ---> optimum @ 190
############################################################

Maximize 2 * x_1 + 3 * x_2
 such that 
	1 * x_1 + 1 * x_2 <= 8
	2 * x_1 + 1 * x_2 <= 12
	1 * x_1 + 2 * x_2 <= 14

############################################################
(4) Optimized solution: 2 6 0 2 0 ---> optimum @ 22
############################################################

Maximize 10 * x_1 + 15 * x_2
 such that 
	-1 * x_1 + -1 * x_2 <= -1
	1 * x_1 + 2 * x_2 <= 6
	2 * x_1 + 1 * x_2 <= 6

############################################################
(5) Optimized solution: 2 2 3 0 0 ---> optimum @ 50
############################################################

Minimize -12 * x_1 + -16 * x_2
 such that 
	-1 * x_1 + -2 * x_2 <= -40
	-1 * x_1 + -1 * x_2 <= -30

############################################################
(6) Optimized solution:  0  0 20 10 ---> optimum @ 400
############################################################
```

## Alternative Methods to solve LPs
Several variations and alternatives to the II-Phase Simplex method exist: 
* Big-M: adds a cost term to the objective function to drive the artificial variables to zero. 
* LP relaxation: relaxes the constraint that varialbes must be integers, and instead allows solutions to contain rational values.
* branch-and-bound: also uses LP relaxation. Uses tree search with pruning to efficiently explore the (continuous) solution state space.


---
layout: post
title:  "Gram-Schmidt with OpenMP"
date:   2025-07-15 08:30:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## OpenMP Intro
`OpenMP` is an API standard for parallel programming in C/C++ and Fortran on shared memory architectures.
OpenMP makes it easy to parallelize programs on multi-core processors and multi-threaded systems with the use of compiler directives, environment variables, and runtime library routines. The API is an add-on in the compiler (e.g. GCC and Clang), and a wide variety of hardware platforms have OpenMP-compatible compilers.

### Thread Management
OpenMP uses a `master` thread to execute sequential portions of a program. When parallel regions of code are encountered, the master thread `forks` into a team of threads. All the threads within the team share the same resources and can access the same global memory space. After each thread completes its assigned work the threads `join` back together and the master thread resumes executing the program in sequential fashion. Sample C++ code below will illustrate the use of `#Pragma` statements to indicate to the compiler which pieces of code should be parallelized.

### CPU-Bound Algorithms
OpenMP shines for CPU-bound algorithms, which are computations that are limited by the performance characteristics of the CPU as oppossed to disk I/O and network communication. These are algorithms that perform many arithmetic / logical operations such that the throughput scales with the CPU frequency and number of cores. Matrix operations such as multiplication and decomposition are prime candidates for parallelization. We'll explore the performance improvements that can be achieved by parallelizing `Gram-Schmidt` with OpenMP.

## Gram-Schmidt
A common theme on this blog is to illustrate some concept or technology by applying it to an interesting problem. Gram-Schmidt is a classic algorithm for computing a set of $n$ orthogonal vectors from a set of vectors that span the space $\mathbb{R}^n$. The set of orthogonal vectors is constructed iteratively by removing components of each vector that are parallel to members of the orthogonal set. This is done by considering 
the `projection` of one vector onto another:

$$
proj_u(v) = \dfrac{\left\langle v, u \right\rangle}{\left\langle u, u \right\rangle} u
$$

where $\left\langle \cdot, \cdot \right\rangle$ indicates the inner product of two vectors.
Intuitively, this expression is taking the component of vector $v$ that is parallel to $u$. This is the 
orthogonal projection of $v$ onto the line spanned by $u$.

![ortho](/images/orthogonal_projection.png)
*<medium>(Fig) Illustration of orthogonal projection of vector $u$ onto $v$ (dashed blue arrow). </medium>*

The Gram-Schmidt process operates on a set of vectors $V=\lbrace v_1, v_2, ..., v_n \rbrace$, iteratively making all vectors in the subset $V'=\lbrace v_{i+1}, ..., v_n \rbrace$
orthogonal to the basis vectors indexed $0-i$. This is done by removing the components of the vectors in $V'$ that are parallel to the basis vectors.
The (orthonormal) basis set is started with vector $u_1 = \dfrac{v_1}{||v_1||}$, and at each iteration $j$ the orthonormalized vector $u_j$ is added to the basis set after $v_j$ has been made orthogonal to all vectors in the basis:

$$
v_j = v_j - \sum_{i=0}^{j-1} \text{proj}_{u_i}(v_j)
$$

We will implement the `modified Gram-Schmidt` algorithm, where at each iteration we project each vector onto the most recently othonormalized vector, thus making each vector orthogonal to this vector. This will give us a nested for-loop structure that will allow us to parallelize the inner loop.

The update equation for each vector $k$ at iteration $i$ looks like this:

$$
v_k^{(i)} = v_k^{(i-1)} - \text{proj}_{v_i}(v_k^{(i-1)})
$$

### Code
Okay, so now we have a high level sketch of the algorithm. Let's take a look at how we can leverage OpenMP to parallelize portions of this algorithm. We know we're going to need some basic linear algebra operations, such as dot project, scalar-vector multiplication, and vector subtraction.

```c++
// Dot project of two vectors.
template <typename T, std::size_t N>
T dot(const std::array<T, N>& a, const std::array<T, N>& b) {
    T res = 0;
    #pragma omp parallel for reduction(+:res)
    for (size_t i = 0; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
```

```c++
// Vector subtraction.
template <typename T, std::size_t N>
std::array<T,N> operator-(const std::array<T,N>& v, const std::array<T,N>& u) {
    std::array<T,N> v_minus_u;
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        v_minus_u[i] = v[i] - u[i];
    }
    return v_minus_u;
}
```

```c++
// Multiply vector by scalar: sv = s * v.
template <typename T, std::size_t N>
std::array<T,N> scale(const std::array<T,N>& v, T s) {
    std::array<T,N> sv;
    std::transform(v.begin(), v.end(), sv.begin(), [&](T x) { return s * x; });
    return sv;
}
```

Notice the compiler directives `#pragma omp parallel for` that instruct the compiler that certain regions of code
should be parallelized. This will spin up the team of threads mentioned earlier, and the total number of iterations in the for-loop will be distributed across the threads. Then each thread will execute in parallel, and there will be a synchronization step before the master thread resumes serial execution. OpenMP offers `scheduling` options for work distribution among threads: `static`, `dynamic`, and `guided`.

The `reduction(+:res)` operation in the `dot` function works by giving each thread its own local copy of the `res` parameter used for summation. Each thread computes its own reduction (in this case a summation) and then the results are combined after each thread has finished executing. This is far simplier and less error prone than an alternative solution of using synchronization features such as `locks` to compute the result, preventing race conditions. OpenMP supports many reduction types, such as `min`, `max`, `+`, `-`, `*`, `|`, `^`, `&&`, `||`, and you can even define custom reductions.

Finally, the code for the `modified Gram-Schmidt` algorithm:
```c++
template <typename T, std::size_t N, std::size_t M>
void MGS(std::array<std::array<T, N>, M>& A) {
    for (size_t i = 0; i < M; ++i) {
        A[i] = scale(A[i], 1. / std::sqrt(dot(A[i], A[i])));
        #pragma omp parallel for
        for (size_t j = i+1; j < M; ++j) {
            // Subtract projection of A[j] onto A[i]
            A[j] = A[j] - scale(A[i], dot(A[j], A[i]));
        }
    }
}
```

### Benchmarking
We benchmark the serial and parallel implementations of the Gram-Schmidt function (`MGS`) by generating a matrix filled with random elements, repeatedly calling `MGS` on this matrix, and computing an average duration.

```c++
template <std::size_t N, std::size_t M>
void benchmark_gsm() {
 std::mt19937 eng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> distr(-1.0, 1.0);
        std::array<std::array<double, N>, M> A;
        for (size_t i = 0; i < M; ++i) {
            std::generate(A[i].begin(), A[i].end(), [&]() {
                return distr(eng);
            });
        }
        assert(!CheckOrthogonality(A));
        auto A_copy = A;
        constexpr size_t num_trials = 20;
        double avg_duration = 0.;
        for (size_t i = 0; i < num_trials; ++i) {
            A = A_copy;
            assert(!CheckOrthogonality(A));
            auto start = std::chrono::high_resolution_clock::now();
            MGS(A);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            avg_duration += duration.count();
            assert(CheckOrthogonality(A));
        }
        avg_duration /= num_trials;
        std::cout << "Avg time: " << avg_duration << " microseconds" << std::endl;
}
```

where we further leverage OpenMP to parallelize the orthogonality check:

```c++
template <typename T, std::size_t N, std::size_t M>
bool CheckOrthogonality(const std::array<std::array<T, N>, M>& A, bool normalized = true) {
    bool orthogonal = true;
    for (size_t i = 0; i < M; ++i) {
        #pragma omp parallel for reduction(&&:orthogonal)
        for (size_t j = i; j < M; ++j) {
            T dp = dot(A[i], A[j]);
            if (j == i) {
                // Check dot product is non-zero (and 1 in orthonormal case)
                if (normalized) {
                    if (std::abs(dp - 1.) > 1e-6) orthogonal = false;
                } else {
                    if (dp <= 0) orthogonal = false;
                }
            } else {
                // Else vectors should be orthogonal
                if (std::abs(dp) > 1e-6) orthogonal = false;
            }
        }
    }
    return orthogonal;
}
```

## Results
![speedup](/images/openmp_speedup.png)
*<medium>(Fig) Time difference (seconds) between parallelization with OpenMP and the serial implementation for running Gram-Schmidt on matrices of different dimensions. </medium>*

In the figure above we see that the OpenMP parallel version of this code starts to achieve speedup after the matrix length/width has at least hundreds of elements. The overhead of thread management and synchronization make the serial implementation faster for smaller matrix sizes. But we start to see a more noticeable difference when the matrices grow to even a medium size, for example $5000 \times 100$ and $500 \times 500$.

## Conclusion
This post provided an introduction to computing with OpenMP. We saw how easy it is to parallelize C++ code with some simple compiler directives. We also saw first hand that there's some overhead required for thread management, and there are many cases where parallelization can actually hurt code performance. OpenMP can work well for many linear algebra operations, and we saw a specific example that combined many linear algebra primitives: Gram-Schmidt. In this application we saw that as the size of our data increases, the performance gains achieved with OpenMP parallelization become very apparent.
---
layout: post
title:  "Experimenting with Central Limit Theorem"
date:   2024-09-09 15:27:43 -0500
categories: jekyll update
---
This is a little demo of the Central Limit Theorem ([CLT][clt]). Simply put, the Central Limit Theorem says that under appropriate conditions the distribution of the sample mean converges to the normal distribution as the size of the sample increases. Given some probability distribution (not necessarily the Normal Distribution), and collect a sufficient number of samples (sources say at minimum 30). The sample mean is defined as the average of this sample population. Repeating this procedure 
many times, you'll notice that the distribution of the sample means is normally distributed.

TODO: insert latex of formal statement?

First we'll create a class that samples from some statistical distribution:
{% highlight C++ %}
#pragma once

#include <random>

namespace stats {
// Samples from some statistical distribution
template <typename T>
class Sampler {
    public:
        Sampler(const T mu);

        // Generates a sample of n elements from the distribution
        std::vector<T> gen(size_t n);

    private:
        std::poisson_distribution<T> distribution_;
        std::mt19937 gen_;
};
{% endhighlight %}

Next we'll implement the constructor and `gen` function that samples from the distribution: 

{% highlight C++ %}
template <typename T>
Sampler<T>::Sampler(const T mu) : distribution_(mu), gen_((std::random_device())()) {}

template <typename T>
std::vector<T> Sampler<T>::gen(size_t n) {
    std::vector<T> samples;
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(distribution_(gen_));
    }
    return samples;
}
}
{% endhighlight %}

The following code snipet generates samples from the distribution and records the mean. The is repeated for various sample sizes:

{% highlight C++ %}
template <typename T> 
double Mean(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0) / static_cast<double>(v.size());
}

int main () {
    constexpr uint32_t mu = 1;
    stats::Sampler sampler(mu);
    const size_t num_iterations = 100;
    for (size_t sample_size = 30; sample_size <= 1e5; sample_size *= 10) {
        std::vector<double> sample_means(num_iterations);
        for (size_t i = 0; i < num_iterations; ++i) {
            auto sample = sampler.gen(sample_size);
            sample_means[i] = Mean(sample);
        }
{% endhighlight %}

Running the simulation and plotting the results with Python we get something like:
![CLT](/images/simulation_results.png)

[clt]:         https://en.wikipedia.org/wiki/Central_limit_theorem

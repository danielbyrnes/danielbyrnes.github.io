---
layout: post
title:  "Experimenting with Central Limit Theorem"
date:   2024-09-09 15:27:43 -0500
categories: jekyll update
---
This is a little demo of the Central Limit Theorem (CLT). Simply put, the Central Limit Theorem says that under appropriate conditions the distribution of the sample mean converges to the normal distribution as the size of the sample increases. Given some probability distribution (not necessarily the Normal Distribution), and collect a sufficient number of samples (sources say at minimum 30). The sample mean is defined as the average of this sample population. Repeating this procedure 
many times, you'll notice that the distribution of the sample means is normally distributed.

TODO: insert latex of formal statement

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

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[clt]:         https://en.wikipedia.org/wiki/Central_limit_theorem
[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

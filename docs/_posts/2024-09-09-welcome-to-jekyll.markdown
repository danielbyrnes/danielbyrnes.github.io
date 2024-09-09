---
layout: post
title:  "Experimenting with Central Limit Theorem"
date:   2024-09-09 15:27:43 -0500
categories: jekyll update
---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

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

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

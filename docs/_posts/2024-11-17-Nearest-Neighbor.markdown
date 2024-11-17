---
layout: post
title:  "Nearest Neighbor Search with KDTrees"
date:   2024-11-17 13:00:00 -0500
categories: jekyll update
---

Searching for a nearest neighbor in a collection of N-dimensional points in Euclidean space amounts to finding the point that minimizes some distance metric with respect to the query point. A brute force solution is linear in the size of the candidate size (the size of the point cloud). The average search time can be decreased to O(logn) using a KDTree. This post discusses a C++ implementation of the KDTree, along with point cloud generation, and identifies some of the pitfalls of a naive implementation.


Starting from the ground up: a point in Euclidean space. Initially my thoughts were limited to 2/3D space (I'm biased).
{% highlight C++ %}
struct Point3D {
    Point3D(double _x, double _y, double _z) : x(_x),y(_y), z(_z) {}
    double x;
    double y;
    double z;
};
{% endhighlight %}
 It's a fair way to get started, but becomes cumbersome and messy to generalize to N-Dimensional space. Eventually I settled on the following implementation of a `point`.

{% highlight C++ %}
namespace geometry {
struct PointND {
    PointND(std::vector<double>&& values) : data(values) {}
    PointND() {}
    std::vector<double> data;

    double& operator[](int index) {
        if (index < 0 || index >= data.size()) {
            throw std::out_of_range("index out of range");
        }
        return data[index];
    }

    double operator[](int index) const {
        if (index < 0 || index >= data.size()) {
            throw std::out_of_range("index out of range");
        }
        return data[index];
    }

    double norm2() {
        double val = 0.;
        for (const auto& v : data) {
            val += v * v;
        }
        return std::sqrt(val);
    }

    friend PointND operator-(const PointND& a, const PointND& b) {
        std::vector<double> difference(a.data.size());
        std::transform(a.data.begin(), a.data.end(), b.data.begin(), difference.begin(), std::minus<double>());
        return PointND(std::move(difference));
    }

    friend std::ostream& operator<<(std::ostream& os, const PointND& p) {
        std::string buff = "(";
        for (const auto& val : p.data) {
            buff += std::to_string(val) + ",";
        }
        // Remove last ","
        buff.pop_back();
        buff += ")";
        os << buff;
        return os;
    }
};
}
{% endhighlight %}

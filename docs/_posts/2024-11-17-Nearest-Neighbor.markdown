---
layout: post
title:  "Nearest Neighbor Search with KDTrees"
date:   2024-11-17 13:00:00 -0500
categories: jekyll update
---

Searching for a nearest neighbor in a collection of N-dimensional points in Euclidean space amounts to finding the point that minimizes some distance metric with respect to the query point. A brute force solution is linear in the size of the candidate size (the size of the point cloud). The average search time can be decreased to O(logn) using a KDTree. This post discusses a C++ implementation of the KDTree, along with point cloud generation, and identifies some of the pitfalls of a naive implementation.

A KDTree is a binary tree that organizes spatial data by comparing points along some axis at each level of the tree. Each node effectively splits points along an axis-aligned hyperplane. K-dimensional points are inserted into the K-D tree by iteratively splitting along each axis. Starting from the root node, the tree is traversed by comparing against the x-axis, the y-axis, etc until a leaf node is reached and the new point can be inserted. Applications of KDTrees include point cloud alignment, where a subroutine is to find the closest points in a reference point cloud. Another application is range-based searching, where it is desired to search for points within sub subrange of Euclidean space.

## Point in N-dimensions
Starting from the ground up: a point in Euclidean space. Initially my thoughts were limited to 2/3D space (I'm biased). A data structure `struct Point3D` with fields `x,y,z`.
 It's a fair way to get started, but becomes cumbersome and messy to generalize to N-Dimensional space. Eventually I settled on the following abstraction of a point.

{% highlight C++ %}
namespace geometry {
struct PointND {
    PointND(std::vector<double>&& values) : data(values) {}
    PointND() {}
    std::vector<double> data;

    // Indexing to access a specific coordinate
    double& operator[](int index) {
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

## Generating the Point Cloud
Generating a point cloud can be done using the uniform real distribution. 
This signifies that the points are equally likely to occur within the bounds of the distribution. 
{% highlight C++ %}
template <uint16_t D>
PointCloud PointCloudFactory<D>::GeneratePointCloud() {
    PointCloud pc;
    pc.reserve(num_points_);
    for (size_t i = 0; i < num_points_; ++i) {
        std::vector<double> coords(D);
        for (size_t i = 0; i < D; ++i) {
            coords[i] = dist_(gen_);
        }
        pc.emplace_back(std::move(coords));
    }
    return pc;
}
{% endhighlight %}


## KDTree
Defining the nodes in the tree as 
{% highlight C++ %}
struct KDNode {
    geometry::PointND point;
    KDNode* left;
    KDNode* right;
};
{% endhighlight %}

and the KDTree interface as 

{% highlight C++ %}
template <int D = 3>
class KDTree {
    public:
        // Default constructor
        KDTree();
        // Instantiate KDTree with all the points up front
        KDTree(const std::vector<geometry::PointND>& points);

        // Public interface to add a point to the tree
        void AddPoint(const geometry::PointND& point);

        // Balances the tree and returns the root node
        KDNode* BalanceTree();
        
        // Remove a node from the tree
        void RemovePoint(const geometry::PointND& point);

        // Find the point in the tree closest to some query point. 
        geometry::PointND FindNearestNeighbor(const KDNode* root, const geometry::PointND& point);
    private:
        // Adds a node to the tree
        KDNode* AddNode(const geometry::PointND& point);

        std::vector<geometry::PointND> points_;
        std::vector<KDNode> nodes_;
};
{% endhighlight %}

# Balancing the tree

After inserting points into the tree it needs to be balanced

{% highlight C++ %}
template <int D>
KDNode* KDTree<D>::BalanceTree() {
    if (points_.empty()) {
        return nullptr;
    }

    // Reserve space for the tree nodes
    nodes_.reserve(points_.size());
    
    std::function<KDNode*(size_t,size_t,int)> build_tree = [&](size_t begin, size_t end, int axis = 0) -> KDNode* {
        if (begin >= end) {
            // invalid search range
            return nullptr;
        }
        auto comp = [&](geometry::PointND a, geometry::PointND b) { 
            return a[axis] < b[axis]; 
        };

        // Split along the median
        size_t n = begin + (end - begin) / 2;
        std::nth_element(points_.begin() + begin, points_.begin() + n, points_.begin() + end, comp);
        axis = (axis + 1) % D;

        auto node_ptr = AddNode(points_[n]);
        node_ptr->left = build_tree(begin, n, axis);
        node_ptr->right = build_tree(n+1, end, axis);
        return node_ptr;
    };

    return build_tree(0, points_.size(), 0);
}
{% endhighlight %}

# Nearest neighbor search
Searching for a nearest point requires traversing through the tree, where branch selection is based on the current axis of comparison
{% highlight C++ %}
template <int D>
geometry::PointND KDTree<D>::FindNearestNeighbor(const KDNode* root, const geometry::PointND& point) {
    const KDNode* ptr = root;
    const KDNode* last_valid_ptr = nullptr;
    int axis = 0;
    constexpr double kDistThreshold = 0.01;
    while (ptr) {
        const double dist = (ptr->point - point).norm2();
        if (dist < kDistThreshold) {
            return ptr->point;
        }
        double df = ptr->point[axis] - point[axis];
        axis = (axis + 1) % D;
        last_valid_ptr = ptr;
        ptr = (df > 0) ? ptr->left : ptr->right;
    }   
    return last_valid_ptr->point;
}
{% endhighlight %}

## Simulation

Generating a point cloud ... 
{% highlight C++ %}
int main() {
    std::cout << "Point Cloud Processing ... " << std::endl;
    constexpr size_t num_points = 100;
    constexpr uint16_t dim = 5u;
    geometry::PointCloudFactory<dim> factory(num_points, 0., 100.);
    auto pc = factory.GeneratePointCloud();
    // Build the KDTree
    graph::KDTree<3> kdtree{};
    for (const auto& point : pc) {
        kdtree.AddPoint(point);
    }
   auto root = kdtree.BalanceTree();
{% endhighlight %}
Randomly select a point from the point cloud and run nearest neighbor search.
{% highlight C++ %}
    std::srand(std::time(0));
    constexpr size_t num_trials = 100;
    for (size_t i = 0; i < num_trials; ++i) {
        // Select random point to search for
        size_t random_point = std::rand() % num_points;
        auto query_point = pc[random_point];
        auto point = kdtree.FindNearestNeighbor(root, query_point);
        assert((point - query_point).norm2() < 0.5);
    }
{% endhighlight %}

## Extensions for Future Work

1. Delete nodes from the tree.
2. Query the r-closest neighbors of any given reference point.
3. Range-based search.

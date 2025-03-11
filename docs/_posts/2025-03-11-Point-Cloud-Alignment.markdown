---
layout: post
title:  "Point Cloud Alignment"
date:   2025-03-11 7:00:00 -0500
categories: jekyll update
---
{% include_relative _includes/mathjax.html %}

## Point Cloud Alignment

Iterative Closest Point (ICP) is a common algorithm for point cloud alignment. Starting with some initial set of correspondences, each inner loop first estimates an orientation that aligns two point clouds, and then recomputes the correspondences between the point clouds given the new orientation that (hopefully) brings correct correspondences closer together.

In this post we will pretend like we have an oracle giving us the correct correspondeces between the point clouds, and instead focus on estimating the orientation that will align the point clouds given the known correspondences. We will then add noise to the correspondences and try to robustly estimate an optimal alignment.

We will rely on some results from linear algebra and 3D geometry, and to support this we'll use the `Eigen` C++ linear algebra library. The explanation presented here is based on [this][procrustes] and [this][ortho_transform] post, which go into much greater detail.

# Math Details

First let's define a distance metric for talking about the closeness of two point clouds after alignment via a rotation transformation.

Define the metric based on the $L2$ error norm: 

$$
E = \sum\limits_{i=0}^n || p_i - R q_i ||_2
$$

where all $p_i$ come from point cloud $P_1$ and $q_i$ from distinct point cloud $P_2$, and $R$ is a rotation matrix.
 
Now assume there exists a rotation matrix that perfectly aligns the point clouds such that the $L2$ distance is 0. Let's find an expression for computing the rotation that minimizes this metric.

Let $M$ be a $3 \times n$ matrix corresponding to point cloud $P_1$, where each column is one of the points in the cloud, and there are $n$ total points. Analogously, define $N$ to be the $3 \times n$ matrix filled column-wise with points from $P_2$. 

Then expaning the error metric above we have that

$$
E = \sum\limits_i (p_i - Rq_i)^T (p_i - Rq_i)
$$

and after distributing factors we see that this expression to equal to 
$$\sum \limits_i ||p_i||_2 + ||q_i||_2 - 2 p_i^T Rq_i$$.

Minimizing $E$ is equivalent to maximizing $p_i^T R q_i$ $\forall i$. All of these terms are found along the diagonal of 
$M^T R N$, and so we want to find $R$ such that 

$$ 
Tr (M^T R N)
$$

is maximized. 

Remembering that the trace of a product of matricies is invariant to product permutations, this is equivalent to maximizing $Tr(RNM^T)$, where $NM^T$ is a $3 \times 3$ matrix. Looking at the SVD of this matrix $$ NM^T = U \Sigma V^T $$ we can work out that the value of $R$ that minimizes $E$ is $R = VU^T$.

# Implementation


{% highlight C++ %}
MatrixXd EstimateOrientation(const MatrixXd& A, const MatrixXd& B) {
  MatrixXd BAt = B * A.transpose();
  Eigen::JacobiSVD<MatrixXd> svd(BAt, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixV() * svd.matrixU().transpose();
}
{% endhighlight %}

To simulate test data we will rotate the point cloud about some randomly generated axis by some angle $\theta$. We'll use the Rodrigues formula to compute the rotation about axis $u$ by angle $\theta$: 

$$R(u,\theta) = cos \theta I + sin \theta \lfloor u \rfloor + (1 - cos \theta) u u^T$$ 

{% highlight C++ %}
MatrixXd RodriguesRotation(const Vector3d& u, double theta) {
  Vector3d u_n = u / u.norm();
  return std::cos(theta) * Eigen::Matrix3d::Identity() + std::sin(theta) * Skew(u_n) + (1 - std::cos(theta)) * u_n * u_n.transpose();
}
{% endhighlight %}
`Skew(u)` here returns the skew symmetric matrix of vector $u$.

To examine the error in our estimated rotations we'll use this formula to compute the angle of a rotation $R$:

$$\theta = \arccos (\frac{(Tr(R) - 1)}{2})$$

{% highlight C++ %}
double AngleBetweenRotations(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb, bool in_degrees = false) {
  Eigen::Matrix3d dR = Ra.transpose() * Rb;
  double arg = (dR.diagonal().sum() - 1.) / 2.;
  arg = std::clamp(arg, -1.0, 1.0);
  double angle = std::acos(arg);
  if (in_degrees) {
    angle *= (180 / M_PI);
  }
  return angle;
}
{% endhighlight %}

We can test our alignment method using synthetic data. This snippet here generates 10 random points
(a $3\times10$ matrix where each column is a point), and a copy to perturb. The copy point cloud is rotated about some randomly generated unit vector and then corrupted with noise. 
We can examine the error by finding the angle between our estimated rotation and the true rotation. We do this incrementally, increasingly rotating more about the same axis to see if there's a relationship between the quality of our estimate and the magnitude of the initial orientation between the point clouds.
{% highlight C++ %}
  Perturbation perturber(0.0, 0.1, 1.0); 
  MatrixXd A = MatrixXd::Random(3,10);
  Eigen::Vector3d u = Eigen::Vector3d::Random();
  for (double theta = 0.1; theta < 0.4; theta += 0.05) {
    // Rotate m and add noise
    Eigen::Matrix3d R = RodriguesRotation(u.normalized(), theta);
    MatrixXd Aprime = (R * A).transpose();
    perturber.Corrupt(Aprime);
    // Estimate rotation between the point clouds
    MatrixXd R_Aprime_to_A = EstimateOrientation(A, Aprime.transpose());
    // Find error angle between estimated and GT rotations
    double angle_radians = AngleBetweenRotations(R, R_Aprime_to_A.transpose(), false);
  }
{% endhighlight %}

You will notice that the error in alignment is within the bounds of the added noise, but doesn't increase
as the orientation offset between the point clouds changes. This gives us confidence that our method should be robust to the magnitude of orientation offset between the two point clouds we want to align, assuming we have correct correspondences.

## Scaling Up (Armadillo Test)
![PLY](/images/armadillos/armadillo_ply.png){:height="360px" width="320px"} ![PC](/images/armadillos/armadillo_view_3.png){:height="400px" width="400px"} 
*<medium>(Left) PLY mesh of Armadillo (Right) Subsampled armadillo point clouds, where one of them has been rotated 30 degrees about the x-axis. </medium>*

We're going to test the point cloud alignment on the armadillo dataset found [here][armadillo]. After generating a rotated and perturbed copy of the point cloud, we'll estimate the orientation that best aligns the two. One issue is that the armadillo point cloud is huge, it has 8648 points! The matrix transpose and multiplication in `EstimateOrientation` alone will be major bottlenecks for making much progress here. Luckily there's no need to use all the points to estimate the orientation. 

Instead we can just sample a small number of correspondences from the armadillos to estimate the orientation. But some correspondences are better than others, given that one of the armadillo point clouds has been corrupted with a small amount of noise. To handle this we can estimate the orientation using a subsample of the correspondences, and then evaluate the quality of the alignment on the entire point cloud. So in short, we can estimate using a subset of our data and then run a much quicker evaluation using all of the data available. This can be done repeatedly such that we'll choose the rotation estimate that best aligns the entire point cloud. This is essentially the [RANSAC][ransac] algorithm for robustly estimating a model given noisy or erroneous data. This step is especially crucial when we also need to estimate the correspondences in full ICP.

We will use this utility class to help us perturb (corrupt) some percentage of the point cloud with some level of error 

{% highlight C++ %}
class Perturbation {
  public:
    /// @brief Perturbation constructor that adds normally distributed random noise to the point cloud.
    /// @param mean The mean of the sampling distribution.
    /// @param stddev The standard deviation of the sampling distribution.
    /// @param percent_corrupted The percent of the point cloud to corrupt [0.-1.].
    Perturbation(double mean, double stddev, double percent_corrupted) : gen_(rd_()) , 
        dist_(std::normal_distribution<>(mean, stddev)), 
        percent_corrupted_(percent_corrupted) {}

    /// @brief  Corrupts the input point cloud (adds noise based on error distribution).
    /// @param pc Input point cloud to perturb.
    /// @return The total amount of error added.
    double Corrupt(MatrixXd& pc) {
      double corruption_level = std::clamp(percent_corrupted_, 0.0, 1.0);
      size_t num_points_corrupted = corruption_level * pc.rows();
      double error = 0.;
      for (size_t i = 0; i < num_points_corrupted; ++i) {
        Eigen::Vector3d noise = dist_(gen_) * Eigen::Vector3d::Random();
        error += noise.norm();
        pc.row(i) += noise;
      }
      return error;
    }

  private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::normal_distribution<> dist_;
    double percent_corrupted_;
};
{% endhighlight %}


Now we can load our armadillo point clouds and repeatedly generate random samples, estimate the orientation between the point clouds, and evaluate the estimate: 
{% highlight C++ %}
  MatrixXd pc = LoadPointCloud(pc_path); // Path to the armadillo point cloud
  MatrixXd pc_prime = LoadPointCloud(pc_prime_path); // Path to a rotated copy of the armadillo
    
  // Corrupt (perturb) the second point cloud with 
  // some normally distributed noise.
  Perturbation perturber(0.0, 0.3, 0.6);
  double corruption_error = perturber.Corrupt(pc_prime);

  // Number of correspondences used to estimate orientation.
  size_t subsample_size = 10;
  MatrixXd sub_pc(3, subsample_size);
  MatrixXd sub_pc_prime(3, subsample_size);
  std::list<size_t> indices(subsample_size);
  std::iota(indices.begin(), indices.end(), 0);

  // Setup the random index generator.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, pc.rows());

  double min_error = std::numeric_limits<double>::max();
  Eigen::AngleAxisd optimal_axis_angle;
  size_t num_trials = 100;
  for (int n = 0; n < num_trials; ++n) {
    // Generate random indices to sample.
    std::vector<size_t> list(subsample_size);
    for (size_t i = 0; i < subsample_size; ++i) {
      list[i] = distrib(gen);
    }

    // Select randomly generated indices from the 
    // point clouds to create the sub-clouds.
    for (size_t index : indices) {
      sub_pc.col(index) = pc.row(list[index]).transpose();
      sub_pc_prime.col(index) = pc_prime.row(list[index]).transpose();
    }

    // Estimate orientation.
    Eigen::Matrix3d R_est = EstimateOrientation(sub_pc, sub_pc_prime);

    // Evaluate quality of estimated rotation.
    double error = 0.;
    for (size_t i = 0; i < pc.rows(); ++i) {
      double pt_error = (pc.row(i).transpose() - R_est * pc_prime.row(i).transpose()).norm();
      error += pt_error;
    }
    error /= pc.rows();
    Eigen::AngleAxisd axisAngle(R_est);
    if (error < min_error) {
      min_error = error;
      optimal_axis_angle = axisAngle;
    }
  }
{% endhighlight %}

![SIMULATION](/images/armadillos/ransac_error.png)
*<medium>Result of RANSAC simulation, where for each trial a subset of correspondences were used
to estimate the orientation, and a global alignment metric was computed. In this simulation the total 
amount of noise added was 0.138387, and the orientation estimate with the lowest error was 29.9908&deg; with an error of 0.143134. </medium>*

## Future Work
In this post we considered the case where we are given the point correspondences we want to align. Full blown ICP estimates both the correspondences AND the orientation to align them. We can compute the correspondences using [Nearest Neighbor Search]({% post_url 2024-11-17-Nearest-Neighbor %}).

[procrustes]: https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/
[ortho_transform]: https://winvector.github.io/xDrift/orthApprox.pdf 
[armadillo]: https://graphics.stanford.edu/data/3Dscanrep/
[ransac]: https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html#RANSAC

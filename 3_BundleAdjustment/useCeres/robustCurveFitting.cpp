#include <ceres/ceres.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <vector>

const double data[] = {
0.000000e+00, 1.133898e+00,
7.500000e-02, 1.334902e+00,
1.500000e-01, 1.213546e+00,
2.250000e-01, 1.252016e+00,
3.000000e-01, 1.392265e+00,
3.750000e-01, 1.314458e+00,
4.500000e-01, 1.472541e+00,
5.250000e-01, 1.536218e+00,
6.000000e-01, 1.355679e+00,
6.750000e-01, 1.463566e+00,
7.500000e-01, 1.490201e+00,
8.250000e-01, 1.658699e+00,
9.000000e-01, 1.067574e+00,
9.750000e-01, 1.464629e+00,
1.050000e+00, 1.402653e+00,
1.125000e+00, 1.713141e+00,
1.200000e+00, 1.527021e+00,
1.275000e+00, 1.702632e+00,
1.350000e+00, 1.423899e+00,
1.425000e+00, 5.543078e+00, // Outlier point
1.500000e+00, 5.664015e+00, // Outlier point
1.575000e+00, 1.732484e+00,
1.650000e+00, 1.543296e+00,
1.725000e+00, 1.959523e+00,
1.800000e+00, 1.685132e+00,
1.875000e+00, 1.951791e+00,
1.950000e+00, 2.095346e+00,
2.025000e+00, 2.361460e+00,
2.100000e+00, 2.169119e+00,
2.175000e+00, 2.061745e+00,
2.250000e+00, 2.178641e+00,
2.325000e+00, 2.104346e+00,
2.400000e+00, 2.584470e+00,
2.475000e+00, 1.914158e+00,
2.550000e+00, 2.368375e+00,
2.625000e+00, 2.686125e+00,
2.700000e+00, 2.712395e+00,
2.775000e+00, 2.499511e+00,
2.850000e+00, 2.558897e+00,
2.925000e+00, 2.309154e+00,
3.000000e+00, 2.869503e+00,
3.075000e+00, 3.116645e+00,
3.150000e+00, 3.094907e+00,
3.225000e+00, 2.471759e+00,
3.300000e+00, 3.017131e+00,
3.375000e+00, 3.232381e+00,
3.450000e+00, 2.944596e+00,
3.525000e+00, 3.385343e+00,
3.600000e+00, 3.199826e+00,
3.675000e+00, 3.423039e+00,
3.750000e+00, 3.621552e+00,
3.825000e+00, 3.559255e+00,
3.900000e+00, 3.530713e+00,
3.975000e+00, 3.561766e+00,
4.050000e+00, 3.544574e+00,
4.125000e+00, 3.867945e+00,
4.200000e+00, 4.049776e+00,
4.275000e+00, 3.885601e+00,
4.350000e+00, 4.110505e+00,
4.425000e+00, 4.345320e+00,
4.500000e+00, 4.161241e+00,
4.575000e+00, 4.363407e+00,
4.650000e+00, 4.161576e+00,
4.725000e+00, 4.619728e+00,
4.800000e+00, 4.737410e+00,
4.875000e+00, 4.727863e+00,
4.950000e+00, 4.669206e+00
};

struct costfunctor {
public:
    costfunctor(cv::Point2d p) : p_(p) {}

    template<typename T>
    bool operator() (const T* const abc, T* residual) const {
        residual[0] = p_.y - exp(abc[0] * p_.x * p_.x + abc[1] * p_.x + abc[2]);
        return true;
    }

private:
    const cv::Point2d p_;

};

int main(int argc, char *argv[])
{
    double a_initial = 0.01, b_initial = 0.02, c_initial = 0.5;
    double a_gt = 0.02, b_gt = 0.03, c_gt = 1;
    double abc[3] = {a_initial, b_initial, c_initial};

    cv::RNG rng;
    std::vector<cv::Point2d> points;

    for (size_t i = 0; i < 1000; i++) {
        double x = i / 1000;
        double y;
        // if (i % 100 == 0) {
        //     y = exp(a_gt * x * x + b_gt * x + c_gt) + rng.gaussian(3.0);
        // }
        // else {
            y = exp(a_gt * x * x + b_gt * x + c_gt) + rng.gaussian(1.0);
        // }
        points.push_back(cv::Point2d(x, y));
    }

    ceres::Problem problem;
    for (size_t i = 0; i < points.size(); i++) {
        ceres::CostFunction* costfunction = 
                    new ceres::AutoDiffCostFunction<costfunctor, 1, 3>(new costfunctor(points[i]));
                    // new ceres::AutoDiffCostFunction<costfunctor, 1, 3>(points[i]); // AutoDiffCostFunction的
                                                                                      // 构造函数参数只能传指向costfunctor的指针
        problem.AddResidualBlock(costfunction, new ceres::CauchyLoss(0.5), abc);
        // problem.AddResidualBlock(costfunction, nullptr, abc);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "optimized a: " << abc[0] << std::endl;
    std::cout << "optimized b: " << abc[1] << std::endl;
    std::cout << "optimized c: " << abc[2] << std::endl;
    
    // ceres::CostFunction costfuncton = 
                // new ceres::AutoDiffCostFunction<costfunctor, 1, 3>(new cos)
    /* code */
    return 0;
}
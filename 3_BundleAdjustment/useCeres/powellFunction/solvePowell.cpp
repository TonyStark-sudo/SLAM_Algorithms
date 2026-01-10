#include <ceres/ceres.h>

#include <cmath>
#include <iostream>

using std::cout;

struct function_1 {
    template <typename T>
    bool operator() (const T* const x1, const T* const x2, T* residual) const {
        residual[0] = x1[0] + T(10) * x2[0];
        return true;
    }
};

struct function_2 {
    template <typename T>
    bool operator() (const T* const x3, const T* const x4, T* residual) const {
        residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
        return true;
    }
};

struct function_3 {
    template <typename T>
    bool operator() (const T* const x2, const T* const x3, T* residual) const {
        residual[0] = (x2[0] - T(2) * x3[0]) * (x2[0] - T(2) * x3[0]);
        return true;
    }
};

struct function_4 {
    template <typename T>
    bool operator() (const T* const x1, const T* const x4, T* residual) const {
        residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);        
        return true;
    }
};

int main(int argc, char const *argv[])
{
    double x1 = 3.0, x2 = -1.0, x3 = 0.0, x4 = 1.0;
    cout << "Before opt x1 = " << x1 
         << ", x2 = " << x2
         << ", x3 = " << x3
         << ", x4 = " << x4 << std::endl;
    ceres::Problem problem;
    ceres::CostFunction* costfunction1 = 
            new ceres::AutoDiffCostFunction<function_1, 1, 1, 1>(new function_1());
    problem.AddResidualBlock(costfunction1, nullptr, &x1, &x2);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<function_2, 1, 1, 1>(new function_2()), nullptr, &x3, &x4);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<function_3, 1, 1, 1>(new function_3()), nullptr, &x2, &x3);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<function_4, 1, 1, 1>(new function_4()), nullptr, &x1, &x4);
    
    ceres::Solver::Options options;
    options.max_linear_solver_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    cout << "after opt x1 = " << x1 
         << ", x2 = " << x2
         << ", x3 = " << x3
         << ", x4 = " << x4 << std::endl;
         
    return 0;
}

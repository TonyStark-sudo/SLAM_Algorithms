#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <chrono>
#include <vector>

using namespace std;

const vector<cv::Point2d> points = {
    cv::Point2d(0.000000e+00, 1.133898e+00),
    cv::Point2d(7.500000e-02, 1.334902e+00),
    cv::Point2d(1.500000e-01, 1.213546e+00),
    cv::Point2d(2.250000e-01, 1.252016e+00),
    cv::Point2d(3.000000e-01, 1.392265e+00),
    cv::Point2d(3.750000e-01, 1.314458e+00),
    cv::Point2d(4.500000e-01, 1.472541e+00),
    cv::Point2d(5.250000e-01, 1.536218e+00),
    cv::Point2d(6.000000e-01, 1.355679e+00),
};

void produceData(vector<cv::Point2d>& dataPoints, double a, double b) {
    double w_sigma = 2;
    double w_inverse = 1.0 / w_sigma;
    cv::RNG rng;

    // dataPoints.clear();

    for (size_t i = 0; i < dataPoints.size(); i++) {
        // double x = static_cast<double>(i / 10);
        double x = static_cast<double>(i + 1) / 10;
        double y = exp(a * x + b) + rng.gaussian(w_sigma * w_sigma);     
        // dataPoints.push_back(cv::Point2d(x, y));
        dataPoints[i] = cv::Point2d(x, y);
    }

}

struct costfunctor {
    costfunctor (cv::Point2d p) : p_(p) {}

    template <typename T>
    // bool operator() (const T* const a, const T* const b, const T* const residual) const {
    bool operator() (const T* const ab, T* residual) const {
        residual[0] = p_.y - exp(ab[0] * p_.x + ab[1]); 
        return true; 
    }

private:
    cv::Point2d p_;
};

int main(int argc, char *argv[])
{
    double a_gt = 0.4, b_gt = 1.0;
    double a_initial = 0.1, b_initial = 0.5;
    vector<cv::Point2d> points(100, cv::Point2d(0, 0));
    double ab[2] = {a_initial, b_initial};

    produceData(points, a_gt, b_gt);

    ceres::Problem problem;
    // for (auto point : points) { // 在这一行中points是按值传递的，如果数据集较大可以通过引用传递来避免不必要的拷贝
    for (const auto& point : points) { 
        ceres::CostFunction* costfunction = 
                new ceres::AutoDiffCostFunction<costfunctor, 1, 2>(new costfunctor(point));
        problem.AddResidualBlock(costfunction, nullptr, ab);    
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;
    cout << "after optimized: \na, b = ";
    for (auto x : ab) cout << x << ", ";
    cout << "\n";

    /* code */
    return 0;
}

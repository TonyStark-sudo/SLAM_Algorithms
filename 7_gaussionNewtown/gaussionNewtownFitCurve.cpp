#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <vector>

using namespace std;

double thres = 0.02;
double initial_a = 0.0;
double initial_b = 0.0;
double initial_c = 0.0;

int iteration = 50;

struct Curve {
    double a;
    double b;
    double c;
};

struct Point {
    double x;
    double y;
};

vector<Point> points;

int main(int argc, char const *argv[])
{
    // 生成带噪声的二次曲线数据 y = 2x^2 + 3x + 1 + 噪声
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 1.0);
    int N = 100;
    points.clear();
    for (int i = 0; i < N; ++i) {
        double x = i * 0.1;
        double y = 2.0 * x * x + 3.0 * x + 1.0 + noise(generator);
        points.push_back({x, y});
    }

    double a = initial_a, b = initial_b, c = initial_c;
    for (int iter = 0; iter < iteration; ++iter) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        double cost = 0;
        for (const auto& point : points) {
            double x = point.x;
            double y = point.y;
            double y_pred = a * x * x + b * x + c;
            double error = y - y_pred;
            cost += error * error;
            // 雅可比
            Eigen::Vector3d J;
            J << -x * x, -x, -1;
            H += J * J.transpose();
            g += -error * J;
        }
        // 求解增量
        Eigen::Vector3d delta = H.ldlt().solve(g);
        a += delta(0);
        b += delta(1);
        c += delta(2);
        if (delta.norm() < thres) {
            break;
        }
        std::cout << "iter " << iter << ": cost = " << cost << ", delta = " << delta.transpose() << std::endl;
    }
    std::cout << "拟合结果: y = " << a << " x^2 + " << b << " x + " << c << std::endl;
    return 0;
}

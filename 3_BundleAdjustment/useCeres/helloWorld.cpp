#include <ceres/ceres.h>
#include <iostream>

// 定义一个结构体表示代价函数，需要重载()来适配ceres的接口
struct CostFunctor {

    // 模版参数T 表示重载的这个函数可以传入多种数值类型
    template <typename T>

    // 第一个const：x 是一个指向 T 类型的指针，表示x指向的内容不能修改
    // 第二个const：指针x本身也是常量，不能让x指向其他的内存地址
    // 第三个const：表示operator()是一个常量成员函数，意味着该成员函数不会修改类的任何成员变量
    bool operator() (const T* const x, T* residual) const {
        residual[0] = (T(10.0) - x[0]);
        return true;
    }

};

int main(int argc, char *argv[])
{
    /* code */
    // 初始化Google日志库，用于调试日志信息
    google::InitGoogleLogging(argv[0]);

    // 待优化变量x的初始化
    double x_initial = 5.0;
    double x = x_initial;

    // 构建寻优问题
    ceres::Problem problem;

    // 实例化了一个名叫AutoDiffCostFunction模版类(该类是继承自CostFunction)的对象，
    // 并将其指针赋给CosFunction*类型的变量
    // AutoDiffCostFunction模版参数<自定义代价函数， 输出残差dim， 优化变量dim>(自定义代价类的实例)
    ceres::CostFunction* costfunction = 
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor()); // 
    problem.AddResidualBlock(costfunction, NULL, &x);

    // 配置运行求解器
    // ceres的Solver类中的Optiona结构体，配置求解器的行为，
    // 例如线性求解器的类型、是否输出优化过程的进展、是否使用特定的求解策略等
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // ceres的Solver类中的Summary结构体，用于存储求解过程的结果和统计信息，
    // 它包含了求解器的状态、迭代次数、最终的残差等信息。
    ceres::Solver::Summary sumery;

    // 调用Solve函数求解优化问题
    ceres::Solve(options, &problem, &sumery);

    std::cout << sumery.BriefReport() << "\n";
    std::cout << "x : " << x_initial << " -> " << x << std::endl;

    return 0;
}


#include "utility.h"

int main(int argc, char *argv[])
{
    std::string path = "./data/poses.txt";
    std::vector<PoseStamped> poses;
    poses = readTumPoses(path);

    for (const auto& pose : poses) {
        std::cout << "p = " << pose.pose.p.transpose() << " q = " << pose.pose.q.coeffs().transpose() << std::endl;
    }
    return 0;
}


#include "iterativeClosestPoint.h"

int main(int argc, char *argv[])
{   
    if (argc != 5) {
        std::cout << "need 4 args: \n"
                  << "Usage: ./build/ICP ../data/1.png ../data/2.png ../data/1_depth.png ../data/2_depth.png\n";
        return -1;
    }

    google::InitGoogleLogging(argv[0]);

    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    cv::Mat depth_1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
    cv::Mat depth_2 = cv::imread(argv[4], cv::IMREAD_UNCHANGED);

    CreateICPProblem createICPProblem(img_1, img_2, depth_1, depth_2);
    std::vector<cv::KeyPoint> kpts1, kpts2;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point3f> points1_3d, points2_3d;

    createICPProblem.find_feature_matches(kpts1, kpts2, matches);
    createICPProblem.get_3D_points(points1_3d, points2_3d, kpts1, kpts2, matches);

    std::cout << "points1_3d nums: " << points1_3d.size() << std::endl;
    std::cout << "points2_3d nums: " << points2_3d.size() << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "id " << i << " point1: ";
        std::cout << "(" << points1_3d[i].x << ", " << points1_3d[i].y << ", " << points1_3d[i].z  << ")" << std::endl;
        std::cout << "id " << i << " point2: ";
        std::cout << "(" << points2_3d[i].x << ", " << points2_3d[i].y << ", " << points2_3d[i].z  << ")" << std::endl;  
    }

    BundleAdjustment icp(points1_3d, points2_3d);
    icp.getInitiValue();
    icp.bundleAdjustment();
    Eigen::Vector3d tran = icp.getTrans();
    Eigen::Quaterniond rota = icp.getRota();

    std::cout << "Translation vector: " << tran.transpose() << std::endl;
    std::cout << "Rotation quaternion: " << rota.coeffs().transpose() << std::endl;
    /* code */
    return 0;
}

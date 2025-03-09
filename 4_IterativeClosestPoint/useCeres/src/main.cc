#include "iterativeClosestPoint.h"

int main(int argc, char *argv[])
{   
    if (argc != 5) {
        std::cout << "need 4 args: \n"
                  << "Usage: ./build/icp ../data/1.png ../data/2.png ../data/1_depth.png ../data/2_depth.png";
        return;
    }
    std::cout << "Creating ICP Problem ...\n";
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

    std::cout << "Creat ICP Problem done !!!\n";
    /* code */
    return 0;
}

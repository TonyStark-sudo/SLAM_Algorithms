#include "iterativeClosestPoint.h"
    
    
void CreateICPProblem::find_feature_matches(std::vector<cv::KeyPoint> &kps_1, 
                                            std::vector<cv::KeyPoint> &kps_2, 
                                            std::vector<cv::DMatch> &matches) {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        cv::Mat des_1, des_2;
        detector->detect(img_1, kps_1);
        detector->detect(img_2, kps_2);
        descriptor->compute(img_1, kps_1, des_1);
        descriptor->compute(img_2, kps_2, des_2);

        std::vector<cv::DMatch> match;
        matcher->match(des_1, des_2, match);
        double min_dist = 10000, max_dist = 0;
        for (auto &m : match) {
            double dist = m.distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        for (auto &m : match) {
            if (m.distance <= std::max(2 * min_dist, 30.0)) {
                matches.push_back(m);
            }
        }
}

std::vector<cv::Point3d> CreateICPProblem::pixel2cam(const std::vector<cv::Point2d>& points_uv, 
                                                     const cv::Mat& K) {
        std::vector<cv::Point3d> points_c;
        for (const auto& point_uv : points_uv) {
            cv::Point3d point_c;
            point_c.x = (point_uv.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
            point_c.y = (point_uv.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
            point_c.z = 1.0;
            points_c.push_back(point_c);
        }
        // for (const auto& m : matches) {
        //     ushort d1 = dpt_1.ptr<unsigned short>(int(kps_1[m.queryIdx].pt.y))[int(kps_1[m.queryIdx].pt.x)];
        //     ushort d2 = dpt_2.ptr<unsigned short>(int(kps_1[m.queryIdx].pt.y))[int(kps_1[m.queryIdx].pt.x)];
        //     if (d1 == 0 || d2 == 0) continue;
        //     double dd1 = double(d1) / 5000;
        //     double dd2 = double(d2) / 5000;
        //     points_c = dd1 * points_c;
        // }

}


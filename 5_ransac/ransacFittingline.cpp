#include <vector>
#include <cmath>
#include <iostream>

struct Line {
    double a;
    double b;
};

struct Point {
    double x;
    double y;
};

double computeDistance(Line line, Point point) {
    return std::abs(line.a * point.x - point.y + line.b) /
           std::sqrt(line.a * line.a + 1);
}

Line ransacFitting(const std::vector<Point>& points, int iteration_count = 1000, 
                   double threshold = 1.0) {
    double bestA, bestB;
    int bestInlierCount = 0;
    for (int i = 0; i < iteration_count; i++) {
        
        // 注意随机取点的方法
        int idx1 = rand() % points.size();
        int idx2 = rand() % points.size();
        if (idx1 == idx2) continue;

        Point p1 = points[idx1], p2 = points[idx2];
        if (p1.x == p2.x) continue;

        Line line_tmp;
        line_tmp.a = (p2.y - p1.y) / (p2.x - p1.x);
        line_tmp.b = p1.y - line_tmp.a * p1.x;

        int inlineCount = 0;
        for (const auto& point : points) {
            double distance = computeDistance(line_tmp, point);
            if (distance < threshold) inlineCount++;
        }

        if (inlineCount > bestInlierCount) {
            bestInlierCount = inlineCount;
            bestA = line_tmp.a;
            bestB = line_tmp.b;
        }       
    }

    return Line{bestA, bestB};
}

int main(int argc, char *argv[])
{
    std::vector<Point> points_noise = {
        {0, 1}, {1, 3}, {2, 5}, {3, 7}, {4, 9}, {5, 11}, {6, 13}, {7, 15}, {8, 17}, {9, 19}, // 直线 y = 2x + 1
        {1, 10}, {3, 1}, {5, 20}, {7, 5}, {9, 25} // 离群点
    };

    Line line = ransacFitting(points_noise);
    std::cout << "Fitting a: " << line.a << "\nFitting b: " << line.b << std::endl;
    /* code */
    return 0;
}

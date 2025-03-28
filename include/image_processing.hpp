#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

struct HoughResult {
    double angle;
    double fixed_angle;
};

class HoughTransform {
public:
    static HoughResult computeAngle(const cv::Mat& image);
    static HoughResult computeAndDisplay(const cv::Mat& image);
};

class CustomCanny {
public:
    static cv::Mat apply(const cv::Mat& image, double lowThreshold, double highThreshold);

private:
    static cv::Mat gaussianBlur(const cv::Mat& image);
    static void sobelGradients(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& direction);
    static cv::Mat nonMaximumSuppression(const cv::Mat& magnitude, const cv::Mat& direction);
    static cv::Mat doubleThreshold(const cv::Mat& suppressed, double lowThreshold, double highThreshold);
    static void hysteresis(cv::Mat& edges, double lowThreshold, double highThreshold);
};

#endif
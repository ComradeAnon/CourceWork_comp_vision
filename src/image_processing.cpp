#include "image_processing.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace cv;
using namespace std;


// 1. Гауссово размытие
Mat CustomCanny::gaussianBlur(const Mat& image) {
    Mat blurred;
    GaussianBlur(image, blurred, Size(5, 5), 1.4);
    return blurred;
}

// 2. Вычисление градиентов (оператор Собеля)
void CustomCanny::sobelGradients(const Mat& image, Mat& magnitude, Mat& direction) {
    Mat gradX, gradY;
    Sobel(image, gradX, CV_64F, 1, 0, 3);
    Sobel(image, gradY, CV_64F, 0, 1, 3);

    magnitude.create(image.size(), CV_64F);
    direction.create(image.size(), CV_64F);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            double gx = gradX.at<double>(y, x);
            double gy = gradY.at<double>(y, x);
            magnitude.at<double>(y, x) = sqrt(gx * gx + gy * gy);
            direction.at<double>(y, x) = atan2(gy, gx);
        }
    }
}

// 3. Подавление немаксимальных пикселей
Mat CustomCanny::nonMaximumSuppression(const Mat& magnitude, const Mat& direction) {
    Mat suppressed = Mat::zeros(magnitude.size(), CV_64F);

    for (int y = 1; y < magnitude.rows - 1; y++) {
        for (int x = 1; x < magnitude.cols - 1; x++) {
            double angle = direction.at<double>(y, x) * 180.0 / CV_PI;
            angle = fmod(angle + 180, 180); // Приведение к 0-180°

            double mag = magnitude.at<double>(y, x);
            double neighbor1 = 0, neighbor2 = 0;

            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                neighbor1 = magnitude.at<double>(y, x - 1);
                neighbor2 = magnitude.at<double>(y, x + 1);
            }
            else if (22.5 <= angle && angle < 67.5) {
                neighbor1 = magnitude.at<double>(y - 1, x + 1);
                neighbor2 = magnitude.at<double>(y + 1, x - 1);
            }
            else if (67.5 <= angle && angle < 112.5) {
                neighbor1 = magnitude.at<double>(y - 1, x);
                neighbor2 = magnitude.at<double>(y + 1, x);
            }
            else {
                neighbor1 = magnitude.at<double>(y - 1, x - 1);
                neighbor2 = magnitude.at<double>(y + 1, x + 1);
            }

            if (mag >= neighbor1 && mag >= neighbor2) {
                suppressed.at<double>(y, x) = mag;
            }
        }
    }
    return suppressed;
}

// 4. Двойной порог
Mat CustomCanny::doubleThreshold(const Mat& suppressed, double lowThreshold, double highThreshold) {
    Mat edges = Mat::zeros(suppressed.size(), CV_8U);

    for (int y = 0; y < suppressed.rows; y++) {
        for (int x = 0; x < suppressed.cols; x++) {
            double mag = suppressed.at<double>(y, x);
            if (mag >= highThreshold) {
                edges.at<uchar>(y, x) = 255; // Сильный пиксель
            }
            else if (mag >= lowThreshold) {
                edges.at<uchar>(y, x) = 128; // Слабый пиксель
            }
        }
    }
    return edges;
}

// 5. Отслеживание краев по гистерезису
void CustomCanny::hysteresis(Mat& edges, double lowThreshold, double highThreshold) {
    for (int y = 1; y < edges.rows - 1; y++) {
        for (int x = 1; x < edges.cols - 1; x++) {
            if (edges.at<uchar>(y, x) == 128) { // Слабый пиксель
                if (edges.at<uchar>(y - 1, x) == 255 || edges.at<uchar>(y + 1, x) == 255 ||
                    edges.at<uchar>(y, x - 1) == 255 || edges.at<uchar>(y, x + 1) == 255 ||
                    edges.at<uchar>(y - 1, x - 1) == 255 || edges.at<uchar>(y - 1, x + 1) == 255 ||
                    edges.at<uchar>(y + 1, x - 1) == 255 || edges.at<uchar>(y + 1, x + 1) == 255) {
                    edges.at<uchar>(y, x) = 255;
                }
                else {
                    edges.at<uchar>(y, x) = 0;
                }
            }
        }
    }
}

// Основная функция
Mat CustomCanny::apply(const Mat& image, double lowThreshold, double highThreshold) {
    Mat gray, blurred, magnitude, direction, suppressed, edges;

    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }
    blurred = gaussianBlur(gray);
    sobelGradients(blurred, magnitude, direction);
    suppressed = nonMaximumSuppression(magnitude, direction);
    edges = doubleThreshold(suppressed, lowThreshold, highThreshold);
    hysteresis(edges, lowThreshold, highThreshold);

    return edges;
}

HoughResult HoughTransform::computeAngle(const Mat& image) {
    Mat gray, edges;

    edges = CustomCanny::apply(image, 50, 150); // Используем наш Canny

    vector<double> testedAngles;
    for (double angle = 84.0; angle <= 96.0; angle += 1.0) {
        testedAngles.push_back(angle);
    }

    vector<double> theta;
    for (double angle : testedAngles) {
        theta.push_back(angle * CV_PI / 180.0);
    }

    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 100, 0, 0, theta.front(), theta.back());

    vector<double> angles;
    for (size_t i = 0; i < lines.size(); i++) {
        angles.push_back(lines[i][1] * 180.0 / CV_PI);
    }

    if (angles.empty()) {
        return { 0, 0 }; // Если не найдено линий, вернуть 0
    }

    double sumAngles = accumulate(angles.begin(), angles.end(), 0.0);
    double meanAngle = sumAngles / angles.size();

    double fixed_angle = -(90.0 - meanAngle);

    return { meanAngle, fixed_angle };
}


HoughResult HoughTransform::computeAndDisplay(const Mat& image) {
    HoughResult result = computeAngle(image);

    // === 1. Предобработка изображения ===
    Mat gray, blurred, edges;
    edges = CustomCanny::apply(image, 50, 150);          // Canny-детектор границ

    // === 2. Применение преобразования Хафа ===
    Mat displayImage;
    cvtColor(edges, displayImage, COLOR_GRAY2BGR); // Делаем цветное изображение для рисования линий
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 100);

    // Ограничиваем количество отображаемых линий (например, 5)
    int maxLines = 10;
    int numLines = min((int)lines.size(), maxLines);

    for (int i = 0; i < numLines; i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(displayImage, pt1, pt2, Scalar(0, 0, 255), 2); // Красные линии на Canny
    }

    imshow("Edges with Hough Lines", displayImage); // Отображаем линии на Canny
    waitKey(0);

    return result;
}
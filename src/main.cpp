#include "image_processing.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>
#include <windows.h>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Функция для чтения истинного угла (инвертируем знак)
double readTrueAngle(const string& filepath) {
    ifstream file(filepath);
    double angle;
    if (file >> angle) {
        return -angle;
    }
    return 0.0;
}

// Средняя абсолютная ошибка (MAE)
double computeMAE(const vector<double>& predicted, const vector<double>& actual) {
    double sumError = 0.0;
    int n = predicted.size();
    for (int i = 0; i < n; i++) {
        sumError += abs(predicted[i] - actual[i]);
    }
    return sumError / n;
}

// Среднеквадратичная ошибка (MSE)
double computeMSE(const vector<double>& predicted, const vector<double>& actual) {
    double sumError = 0.0;
    int n = predicted.size();
    for (int i = 0; i < n; i++) {
        double error = predicted[i] - actual[i];
        sumError += error * error;
    }
    return sumError / n;
}

// Корень из среднеквадратичной ошибки (RMSE)
double computeRMSE(const vector<double>& predicted, const vector<double>& actual) {
    return sqrt(computeMSE(predicted, actual));
}

// Максимальная ошибка (Max Error)
double computeMaxError(const vector<double>& predicted, const vector<double>& actual) {
    double maxError = 0.0;
    int n = predicted.size();
    for (int i = 0; i < n; i++) {
        maxError = max(maxError, abs(predicted[i] - actual[i]));
    }
    return maxError;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001);

    // Проверяем, что переданы два аргумента (пути к изображениям и меткам)
    if (argc < 3) {
        cout << "Использование: program.exe <путь_к_изображениям> <путь_к_labels>" << endl;
        return 1;
    }

    string imageDir = argv[1];  // Путь к папке с изображениями
    string anglesDir = argv[2]; // Путь к папке с метками

    vector<double> predictedAngles;
    vector<double> trueAngles;

    for (const auto& entry : fs::directory_iterator(imageDir)) {
        string imagePath = entry.path().string();
        string filename = entry.path().stem().string() + ".txt";
        string anglePath = anglesDir + "/" + filename;

        if (!fs::exists(anglePath)) {
            cout << "Файл угла не найден: " << anglePath << endl;
            continue;
        }

        Mat image = imread(imagePath);
        if (image.empty()) {
            cout << "Ошибка загрузки изображения: " << imagePath << endl;
            continue;
        }

        // Вычисляем предсказанный угол
        HoughTransform detector;
        HoughResult result = detector.computeAngle(image);
        double predictedAngle = result.fixed_angle;

        // Истинное значение угла
        double trueAngle = readTrueAngle(anglePath);

        predictedAngles.push_back(predictedAngle);
        trueAngles.push_back(trueAngle);
    }

    // Вычисляем метрики
    if (!predictedAngles.empty()) {
        double mae = computeMAE(predictedAngles, trueAngles);
        double mse = computeMSE(predictedAngles, trueAngles);
        double rmse = computeRMSE(predictedAngles, trueAngles);
        double maxError = computeMaxError(predictedAngles, trueAngles);

        cout << "=== Оценка качества алгоритма ===" << endl;
        cout << "Средняя абсолютная ошибка (MAE): " << mae << " градусов" << endl;
        cout << "Среднеквадратичная ошибка (MSE): " << mse << " градусов^2" << endl;
        cout << "Корень из MSE (RMSE): " << rmse << " градусов" << endl;
        cout << "Максимальная ошибка (Max Error): " << maxError << " градусов" << endl;
    }
    else {
        cout << "Ошибка: нет обработанных изображений!" << endl;
    }

    return 0;
}

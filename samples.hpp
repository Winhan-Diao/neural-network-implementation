#pragma once
#include <fstream>
#include "network.hpp"
#include "mnist.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "utils.hpp"
#include <float.h>

using namespace std::literals;

inline void mnist() {
    Network n(28*28, 10, std::vector{50, 40, 30}, std::vector{ActivationFunctions::LEAKYRELU, ActivationFunctions::LEAKYRELU, ActivationFunctions::LEAKYRELU}, ActivationFunctions::SOFTMAX, LossFunctions::CROSS_ENTROPY_LOSS);
    if (std::ifstream ifs{"mnist-network-v2.dat", std::ios::binary}) {
        std::cout << "train on an existing network" << "\r\n";
        ifs >> n;
    } else {
        std::cout << "new network" << "\r\n";
    }
    std::valarray<double> trainLabels{loadLabels("train-labels.idx1-ubyte"s)};
    std::valarray<std::valarray<double>> trainLabelsClassified{classifyLabels(trainLabels)};
    std::valarray<std::valarray<double>> trainImages{loadImages("train-images.idx3-ubyte"s)};
    std::for_each(std::begin(trainImages), std::end(trainImages), [](std::valarray<double>& v){
        v /= 255;
    });
    std::valarray<double> testLabels{loadLabels("t10k-labels.idx1-ubyte"s)};
    std::valarray<std::valarray<double>> testLabelsClassified{classifyLabels(testLabels)};
    std::valarray<std::valarray<double>> testImages{loadImages("t10k-images.idx3-ubyte"s)};
    std::for_each(std::begin(testImages), std::end(testImages), [](std::valarray<double>& v){
        v /= 255;
    });
    decltype(trainLabelsClassified) trimmedTrainLabelsClassified = trainLabelsClassified[std::slice(0, 60000, 1)];
    decltype(trainImages) trimmedTrainImages = trainImages[std::slice(0, 60000, 1)];
    decltype(testImages) trimmedTestImages = testImages[std::slice(0, 10000, 1)];
    decltype(testLabels) trimmedTestLabels = testLabels[std::slice(0, 10000, 1)];

    size_t batchSize = 1;
    train(n, trimmedTrainImages, trimmedTrainLabelsClassified, .0001, 5, batchSize, testImages, testLabels, [](const std::valarray<double>& predicted, const double& actual){
        return getGreatestLabel(predicted) == actual;
    });

    if (std::ofstream ofs{"mnist-network-v2.dat", std::ios::binary}) {
        ofs << n;
        std::cout << "the trained network is saved." << "\r\n";
    }

}

inline void $xor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uid(0, 1);
    Network n(2, 1, std::vector<int>{2}, {ActivationFunctions::SIGMOID}, ActivationFunctions::SIGMOID);
    for (ssize_t i = 0; i < 800'000; ++i) {
        double d1 = uid(gen);
        double d2 = uid(gen);
        auto r = n.train({d1, d2}, {double(d1 != d2)}, .001);
        if (i % 1000 == 0) {
            printValarray(r, std::to_string(i) + ": <"s + std::to_string(d1) + ", "s + std::to_string(d2) + "> "s);
        }
    }
    if (std::ofstream ofs{"xor.dat", std::ios::binary | std::ios::trunc}) {
        ofs << n;
        std::cout << "network saved" << "\r\n";
    }

}

inline void $xorClass() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uid(0, 1);
    Network n(2, 2, std::vector<int>{8, 8}, {ActivationFunctions::LEAKYRELU, ActivationFunctions::LEAKYRELU}, ActivationFunctions::SOFTMAX);
    for (ssize_t i = 0; i < 800'000; ++i) {
        double d1 = uid(gen);
        double d2 = uid(gen);
        auto r = n.train({d1, d2}, {double(d1 == d2), double(d1 != d2)}, .0001);
        if (i % 1000 == 0) {
            printValarray(r, std::to_string(i) + ": <"s + std::to_string(d1) + ", "s + std::to_string(d2) + "> "s);
        }
    }
    if (std::ofstream ofs{"xor.dat", std::ios::binary | std::ios::trunc}) {
        ofs << n;
        std::cout << "network saved" << "\r\n";
    }

}

inline void classify() {
    Network n(1, 10, std::vector<ssize_t>{50, 50}, {ActivationFunctions::PRRELU, ActivationFunctions::PRRELU}, ActivationFunctions::SOFTMAX);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution uid(0, 9);
    for (ssize_t i = 0; i < 200'000; ++i) {
        int d1 = uid(gen);
        std::valarray<double> d2(10);
        d2[d1] = 1;
        auto r = n.train({d1 / 10.}, d2, .001);
        if (i % 1000 == 0) {
            std::cout << "<" << d1 << "> ";
            printValarray(r, "train" + std::to_string(i));
        }
    }

}

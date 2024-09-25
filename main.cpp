#include <fstream>
#include "network.hpp"
#include "mnist.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"

using namespace std::literals;

int main() {
    Network n(28*28, 10, std::vector{100, 100});
    if (std::ifstream ifs{"mnist-network-v1.dat", std::ios::binary}) {
        std::cout << "train on an existing network" << "\r\n";
        ifs >> n;
    } else {
        std::cout << "new network" << "\r\n";
    }
    std::valarray<double> trainLabels{loadLabels("train-labels.idx1-ubyte"s)};
    std::valarray<std::valarray<double>> trainLabelsClassified{classifyLabels(trainLabels)};
    std::valarray<std::valarray<double>> trainImages{loadImages("train-images.idx3-ubyte"s)};
    std::for_each(std::begin(trainImages), std::end(trainImages), [](std::valarray<double>& v){
        return v / 255;
    });
    int correctCounts = 0;
    for (int i = 0; i < trainLabels.size(); ++i) {
        auto r = n.train(trainImages[i], trainLabelsClassified[i], .0001);
        if (getGreatestLabel(r) == static_cast<int>(trainLabels[i]))
            ++correctCounts;
        if (i % 100 == 0) {
            std::cout << "<acc: " << (correctCounts / 100.) << "> ";
            std::cout << "<" << trainLabels[i] << "> ";    
            printValarray(r, "train"s + std::to_string(i));
            correctCounts = 0;
        }
    }
    if (std::ofstream ofs{"mnist-network-v1.dat", std::ios::binary}) {
        ofs << n;
        std::cout << "the trained network is saved." << "\r\n";
    }

}

void archived1() {
    Network n(2, 2, std::vector{128});
    std::valarray<double> r1 = n.train({1, 4}, {0, 1}, .1);
    std::valarray<double> r2 = n.train({5, 2}, {1, 0}, .1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution urd(0., 10.);
    for (ssize_t j =0; j < 100; ++j) {
        for (ssize_t i = 0; i < 500; ++i) {
                auto d1 = urd(gen);
                auto d2 = urd(gen);
                n.train({d1, d2}, {double(d1 > d2), double(d2 > d1)}, .1);
        }
    }
    // std::valarray<double> r3 = n.train({1, 2}, {0, 1}, 0);
    
    printValarray(r1, "result1");
    printValarray(r2, "result2");
    printValarray(n.train({1, 2}, {0, 1}, 0), "result3");
    printValarray(n.train({3, 2}, {1, 0}, 0), "result4");
    printValarray(n.train({2, 1}, {1, 0}, 0), "result5");
    printValarray(n.train({7, 1}, {1, 0}, 0), "result6");
    printValarray(n.train({0, 9}, {0, 1}, 0), "result7");
    printValarray(n.train({5, 2}, {1, 0}, 0), "result8");

}

void archived2() {
    Network n(1, 10, std::vector<ssize_t>{50, 50});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution uid(0, 9);
    for (ssize_t i = 0; i < 200000; ++i) {
        int d1 = uid(gen);
        std::valarray<double> d2(10);
        d2[d1] = 1;
        auto r = n.train({d1 / 10.}, d2, .12);
        std::cout << "<" << d1 << "> ";
        printValarray(r, "train" + std::to_string(i));
    }
    // printValarray(n.inputLayer.weights, "n.inputLayer.weights"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].weights, "n.hiddenLayers["s + std::to_string(i) + "].weights"s);
    // printValarray(n.outputLayer.weights, "n.outputLayer.weights");
    // printValarray(n.inputLayer.biases, "n.inputLayer.biases"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].biases, "n.hiddenLayers["s + std::to_string(i) + "].biases"s);
    // printValarray(n.outputLayer.biases, "n.outputLayer.biases");

    printValarray(n.train({.9}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 0));
    printValarray(n.train({.8}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, 0));
    printValarray(n.train({.7}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, 0));
    printValarray(n.train({.6}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, 0));
    printValarray(n.train({.5}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, 0));
    printValarray(n.train({.4}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, 0));
    printValarray(n.train({.3}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, 0));
    printValarray(n.train({.2}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, 0));
    printValarray(n.train({.1}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0));
    printValarray(n.train({.0}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0));
}

void archived3() {
    Network n(2, 2, std::vector{8, 8});
    std::valarray<double> r1 = n.train({1, 4}, {0, 1}, .1);
    std::valarray<double> r2 = n.train({5, 2}, {1, 0}, .1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution uid(0, 1);
    for (ssize_t i =0; i < 200'000; ++i) {
        double d1 = uid(gen);
        double d2 = uid(gen);
        auto r = n.train({d1, d2}, {double(d1 == d2), double(d2 != d1)}, .005);
        std::cout << "<" << d1 << ", " << d2 << "> ";
        printValarray(r, "train" + std::to_string(i));
    }
    // printValarray(n.inputLayer.weights, "n.inputLayer.weights"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].weights, "n.hiddenLayers["s + std::to_string(i) + "].weights"s);
    // printValarray(n.outputLayer.weights, "n.outputLayer.weights");
    // printValarray(n.inputLayer.biases, "n.inputLayer.biases"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].biases, "n.hiddenLayers["s + std::to_string(i) + "].biases"s);
    // printValarray(n.outputLayer.biases, "n.outputLayer.biases");

}

void archived4() {
    Network n(28*28, 10, std::vector{128, 128});
    std::valarray<double> trainLabels{loadLabels("train-labels.idx1-ubyte"s)};
    std::valarray<std::valarray<double>> trainLabelsClassified{classifyLabels(trainLabels)};
    std::valarray<std::valarray<double>> trainImages{loadImages("train-images.idx3-ubyte"s)};
    std::for_each(std::begin(trainImages), std::end(trainImages), [](std::valarray<double> v){
        return v / 255;
    });
    for (int i = 0; i < trainLabels.size(); ++i) {
        std::cout << "<" << trainLabels[i] << "> ";
        printValarray(n.train(trainImages[i], trainLabelsClassified[i], .01), "train"s + std::to_string(i));
    }

}

void mnist1() {
    Network n(28*28, 10, std::vector{100, 100});
    std::valarray<double> trainLabels{loadLabels("train-labels.idx1-ubyte"s)};
    std::valarray<std::valarray<double>> trainLabelsClassified{classifyLabels(trainLabels)};
    std::valarray<std::valarray<double>> trainImages{loadImages("train-images.idx3-ubyte"s)};
    std::for_each(std::begin(trainImages), std::end(trainImages), [](std::valarray<double>& v){
        return v / 255;
    });
    // for (int i = 0; i < 10; ++i) {
    //     printImage(trainImages[i]);
    //     std::cout << trainLabels[i] << "\r\n";
    // }
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < trainLabels.size(); ++i) {
            auto r = n.train(trainImages[i], trainLabelsClassified[i], .0001);
            if (i % 100 == 0) {
                std::cout << "<" << trainLabels[i] << "> ";    
                printValarray(r, "train"s + std::to_string(i));
            }
        }

    // printValarray(n.inputLayer.weights, "n.inputLayer.weights"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].weights, "n.hiddenLayers["s + std::to_string(i) + "].weights"s);
    // printValarray(n.outputLayer.weights, "n.outputLayer.weights");
    // printValarray(n.inputLayer.biases, "n.inputLayer.biases"s);
    // for (ssize_t i = 0; i < n.hiddenLayers.size(); ++i)
    //     printValarray(n.hiddenLayers[i].biases, "n.hiddenLayers["s + std::to_string(i) + "].biases"s);
    // printValarray(n.outputLayer.biases, "n.outputLayer.biases");

}

void archived5() {
    Network n(2, 2, std::vector{8, 8});
    std::valarray<double> r1 = n.train({1, 4}, {0, 1}, .1);
    std::valarray<double> r2 = n.train({5, 2}, {1, 0}, .1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution uid(0, 1);
    for (ssize_t i = 0; i < 20'000; ++i) {
        double d1 = uid(gen);
        double d2 = uid(gen);
        auto r = n.train({d1, d2}, {double(d1 == d2), double(d2 != d1)}, .05);
        std::cout << "<" << d1 << ", " << d2 << "> ";
        printValarray(r, "train" + std::to_string(i));
    }
    std::cout << n;

}

void misc1() {
    Layer l(2, 5, std::mt19937(std::random_device()()), std::normal_distribution(0., .7), std::normal_distribution(0., .001), ActivationFunctions::RELU, LossFunctions::CROSS_ENTROPY_LOSS);
    Layer l2(10, 10);
    std::string s;
    std::ofstream ofs("out_test.txt", std::ios::binary);
    ofs << l;
    ofs.flush();
    std::ifstream ifs("out_test.txt");
    ifs >> l2;
    std::cout << l2;
}

void misc2() {
    Network n(5, 8, std::vector{20, 10});
    if (std::ofstream ofs{"out_test1.txt", std::ios::binary}) {
        ofs << n;
    }
    Network n2(6, 3, std::vector{2, 3, 3, 3});
    if (std::ifstream ifs{"out_test1.txt", std::ios::binary}) {
        ifs >> n2;
    }
    std::cout << n2;

}
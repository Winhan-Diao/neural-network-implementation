#pragma once
#include <valarray>
#ifndef M_PI
    #define M_PI 3.1415926535897932384626433832795
#endif

enum class LossFunctions {
    MSE,
    CROSS_ENTROPY_LOSS,
    CROSS_ENTROPY_LOSS_V2,
    TAN,
    POLICY_GRADIENT_LOSS,
    CUSTOM,
};

struct LossFunction {
    virtual std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) = 0;
    virtual double operator() (double actual, double predicted) = 0;
};

struct MSE: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        return predicted - actual;
    }
    double operator() (double actual, double predicted) override {
        return predicted - actual;
    }
};

struct CrossEntropyLoss: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (actual[i] == 0)? (-1. / (predicted[i] - 1.)): (-1. / predicted[i]);  
        }
        return loss;
    }
    double operator() (double actual, double predicted) override {
        return (actual == 0)? (-1. / (predicted - 1.)): (-1. / predicted);
    }
};

struct CrossEntropyLossV2: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (actual[i] == 0)? (-1. / (predicted[i] - 1.01) - .99): (-1. / (predicted[i] + .01) + .99);  
        }
        return loss;
    }
    double operator() (double actual, double predicted) override {
        return (actual == 0)? (-1. / (predicted - 1.01) - .99): (-1. / (predicted + .01) + .99);
    }
};

struct Tan: LossFunction {
    static constexpr double tau = M_PI / 2; 
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (std::pow(std::tan(((actual[i] == 0)? predicted[i]: 1. - predicted[i]) * tau), 2) + 1) * tau * ((actual[i] == 0)? 1: -1);
        }
        return loss;
    }
    double operator() (double actual, double predicted) override {
        return (std::pow(std::tan(((actual == 0)? predicted: 1. - predicted) * tau), 2) + 1) * tau * ((actual == 0)? 1: -1);
    }
};

struct PolicyGradientLoss: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& q, const std::valarray<double>& predicted) override {
        return -q / (predicted + 1e-10);
    }
    double operator() (double q, double predicted) override {
        return -q / (predicted + 1e-10);
    }
};

struct Custom: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& loss, const std::valarray<double>& predicted) override {
        return loss;
    }
    double operator() (double loss, double predicted) override {
        return loss;
    }
};


static std::unique_ptr<LossFunction> buildLossFunction(const LossFunctions& n) {
    switch (n) {
        case LossFunctions::CROSS_ENTROPY_LOSS:
            return std::make_unique<CrossEntropyLoss>();
        case LossFunctions::CROSS_ENTROPY_LOSS_V2:
            return std::make_unique<CrossEntropyLossV2>();
        case LossFunctions::MSE:
            return std::make_unique<MSE>();
        case LossFunctions::TAN:
            return std::make_unique<Tan>();
        case LossFunctions::POLICY_GRADIENT_LOSS:
            return std::make_unique<PolicyGradientLoss>();
        case LossFunctions::CUSTOM:
            return std::make_unique<Custom>();
        default:
            throw std::runtime_error{"cannot build LossFunction"};
    }
}

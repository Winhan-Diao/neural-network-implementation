#pragma once
#include <valarray>

enum class LossFunctions {
    MSE,
    CROSS_ENTROPY_LOSS,
};

struct LossFunction {
    virtual std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) = 0;
    static std::unique_ptr<LossFunction> buildLossFunction(const LossFunctions&);
};

struct MSE: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
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
};

std::unique_ptr<LossFunction> LossFunction::buildLossFunction(const LossFunctions& n) {
    switch (n) {
        case LossFunctions::CROSS_ENTROPY_LOSS:
            return std::make_unique<CrossEntropyLoss>();
        case LossFunctions::MSE:
            return std::make_unique<MSE>();
        default:
            throw std::runtime_error{"cannot build LossFunction"};
    }
}

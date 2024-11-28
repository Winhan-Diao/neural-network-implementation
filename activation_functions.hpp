#pragma once
#include <valarray>
#include <functional>
#include "traits.hpp"
#define EXP_700_ 1.0142320547350045094553295952313e+304
#define EXP_N700_ 9.8596765437597708567053729478495e-305

using namespace std::literals;

// template <template <typename> typename T, typename V,  class U = decltype("valarr"s)>
// static void printValarray(const T<V>& valarr, U&& name = "valarr"s) {
//     std::cout << name << ": {"s;
//     std::for_each(std::cbegin(valarr), std::cend(valarr), [&valarr](const auto& a){
//         if constexpr (is_valarray<V>::value) {
//             printValarray(a, "innerVArr"s);
//         } else {
//             std::cout << a;
//         }
//         std::cout << ((&a == std::cend(valarr) - 1)? ""s : ", "s);
//     });
//     std::cout << "}"s << "\r\n"s;
// }

enum class ActivationFunctions {
    INVALID,
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    PRRELU,
    SOFTMAX,
    STABLE_SOFTMAX,
    STABLE_SOFTMAX_V3,
    TAYLOR_SOFTMAX,
    CUBEROOT,
    SGNEXP,
};

struct ActivationFunction {
    virtual std::valarray<double> operator() (const std::valarray<double>& x) = 0;
    virtual std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) = 0;
};

struct Sigmoid: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return 1 / (1 + std::exp(-x));
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return y * (1 - y) * usGrad;
    }
};

struct Tanh: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return std::tanh(x);
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (1 - y * y) * usGrad;
    }
};

struct Relu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return std::max(0.0, e);
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return e > 0 ? 1.0 : 0.0;
        });
        return r * usGrad;
    }
};

struct LeakyRelu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return (e > 0)? e: .02 * e;
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return e > 0 ? 1.0 : 0.02;
        });
        return r * usGrad;
    }
};

struct PrRelu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return (e > 0)? e: .2 * e;
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return (e > 0)? 1. : .2;
        });
        return r * usGrad;
    }
};

struct Softmax: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto expX = std::exp(x);
        double expSum = expX.sum();
        return std::move(expX) / expSum;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        // double sum = (y * usGrad).sum();
        // return -y * (sum - usGrad);                                                      // wrong
        return y * (y.sum() * usGrad - (y * usGrad).sum());
    }
};

struct StableSoftmax: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto expX = std::exp(x - x.max());
        double expSum = expX.sum();
        return std::move(expX) / expSum;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {                                                     // wrong
        return y * (y.sum() * usGrad - (y * usGrad).sum());
        // return (y.sum() * usGrad - (y * usGrad).sum()) / std::pow(y.sum(), 2) * y;       // redundant
    }
};

struct[[deprecated]] StableSoftmaxV2: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto clampedX = x.apply([](double d) ->double { return std::clamp(d, -700., 700.); });
        auto expX = std::exp(clampedX - clampedX.max());
        double expSum = expX.sum();
        return std::move(expX) / expSum;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (y * (y.sum() * usGrad - (y * usGrad).sum())).apply([](double d) ->double { return (d > EXP_700_ || d < EXP_N700_)? 0.0: d; });
        // return (y.sum() * usGrad - (y * usGrad).sum()) / std::pow(y.sum(), 2) * y;       // redundant
    }
};

struct StableSoftmaxV3: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto clampedX = std::tanh(x / 200) * 200;
        auto expX = std::exp(clampedX - clampedX.max());
        double expSum = expX.sum();
        return std::move(expX) / expSum;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        // printValarray(usGrad);
        return y * (y.sum() * usGrad - (y * usGrad).sum()) * (-std::pow(std::log(y) / 200, 2) + 1);
        // return (y.sum() * usGrad - (y * usGrad).sum()) / std::pow(y.sum(), 2) * y;       // redundant
    }
};

struct[[deprecated]] TaylorSoftmax: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto taylor = 0.5 * std::pow(x, 2) + x + 1;
        double taylorSum = taylor.sum();
        return std::move(taylor) / std::move(taylorSum);
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (y.sum() * usGrad - (y * usGrad).sum()) / std::pow(y.sum(), 2) * std::sqrt(2 * y - 1);
    }
};

struct CubeRoot: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return std::pow(std::abs(x), 1. / 3) * x.apply([](double v) -> double { return v < 0? -1: 1; });
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return usGrad / (y * y * 3 + 1e-5);
    }
};

struct SgnExp: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return (1 - std::exp(-std::abs(x))) * x.apply([](double v) -> double { return v < 0? -1: 1; });
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (1 - std::abs(y)) * usGrad;
    }
};

static std::unique_ptr<ActivationFunction> buildActivationFunction(const ActivationFunctions& n) {
    switch (n) {
        case ActivationFunctions::SIGMOID:
            return std::make_unique<Sigmoid>();
        case ActivationFunctions::TANH:
            return std::make_unique<Tanh>();
        case ActivationFunctions::RELU:
            return std::make_unique<Relu>();
        case ActivationFunctions::LEAKYRELU:
            return std::make_unique<LeakyRelu>();
        case ActivationFunctions::PRRELU:
            return std::make_unique<PrRelu>();
        case ActivationFunctions::SOFTMAX:
            return std::make_unique<Softmax>();
        case ActivationFunctions::STABLE_SOFTMAX:
            return std::make_unique<StableSoftmax>();
        case ActivationFunctions::STABLE_SOFTMAX_V3:
            return std::make_unique<StableSoftmaxV3>();
        case ActivationFunctions::TAYLOR_SOFTMAX:
            return std::make_unique<TaylorSoftmax>();
        case ActivationFunctions::CUBEROOT:
            return std::make_unique<CubeRoot>();
        case ActivationFunctions::SGNEXP:
            return std::make_unique<SgnExp>();
        case ActivationFunctions::INVALID:
        default:
            throw std::runtime_error{"cannot build ActivationFunction"};
    }
}

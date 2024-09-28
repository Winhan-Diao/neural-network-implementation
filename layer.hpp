#pragma once
#include <valarray>
#include <functional>
#include <random>
#include <cassert>
#include <memory>
#include <charconv>
#include <cctype>
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "stream_utils.hpp"

using namespace std::literals;

class Layer {
    std::valarray<double> biases;
    std::valarray<std::valarray<double>> weights;
    std::valarray<double> values;
    std::valarray<double> deltas;
    ActivationFunctions activationFunctionEnum;
    std::unique_ptr<ActivationFunction> activationFunction;
    LossFunctions lossFunctionEnum;
    std::unique_ptr<LossFunction> lossFunction;
    ssize_t layerSize;
    ssize_t nextLayerSize;
public:
    template <class T1 = decltype(std::normal_distribution(0., .7)), class T2 = decltype(std::normal_distribution(0., .001))>
    Layer(ssize_t nodeCounts
            , ssize_t nextLayerNodeCounts = 0
            , std::mt19937 gen = std::mt19937(std::random_device()())
            , T1&& weightDistribution = std::normal_distribution(0., .7)
            , T2&& biaseDistribution = std::normal_distribution(0., .001)
            , const ActivationFunctions& activationFunctionEnum = ActivationFunctions::SIGMOID
            , const LossFunctions& lossFunctionEnum = LossFunctions::MSE
        ): 
        biases(nodeCounts), 
        weights(std::valarray<double>(.5, nextLayerNodeCounts), nodeCounts),     //bad perf 
        values(nodeCounts), 
        deltas(nodeCounts),
        activationFunctionEnum(activationFunctionEnum),
        activationFunction(ActivationFunction::buildActivationFunction(activationFunctionEnum)), 
        lossFunctionEnum(lossFunctionEnum),
        lossFunction(LossFunction::buildLossFunction(lossFunctionEnum)),
        layerSize(nodeCounts),
        nextLayerSize(weights.size()? weights[0].size(): 0)
    {
        for (ssize_t i = 0; i < biases.size(); ++i) {
            biases[i] = biaseDistribution(gen);
        }
        for (ssize_t i = 0; i < weights.size(); ++i) {
            for (ssize_t j = 0; j < (weights.size()? weights[0].size(): 0); ++j) {
                weights[i][j] = weightDistribution(gen);
            }
        }
    }
    Layer(const std::valarray<double>& biases
            , const std::valarray<std::valarray<double>>& weights
            , const ActivationFunctions& activationFunctionEnum
            , const LossFunctions& lossFunctionEnum
        ): 
        biases(biases),
        weights(weights),
        values(weights.size()),
        deltas(weights.size()),
        activationFunctionEnum(activationFunctionEnum),
        activationFunction(ActivationFunction::buildActivationFunction(activationFunctionEnum)),
        lossFunctionEnum(lossFunctionEnum),
        lossFunction(LossFunction::buildLossFunction(lossFunctionEnum)),
        layerSize(weights.size()),
        nextLayerSize(weights.size()? weights[0].size(): 0)
    {}
    Layer() = default;
    void forward(const Layer& prevLayer) {
        std::valarray<double> tmpValarr(this->values.size());
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < prevLayer.weights.size(); ++j) {
                tmpValarr[i] += prevLayer.weights[j][i] * prevLayer.values[j];
            }
        }
        this->values = (*activationFunction)(static_cast<std::valarray<double>&&>(this->biases + std::move(tmpValarr)));
    }
    void backward(const Layer& nextLayer, double learningRate) {
        std::valarray<double> upstreamGradients(this->deltas.size());
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                upstreamGradients[i] += nextLayer.deltas[j] * this->weights[i][j];
            }
        }
        this->deltas = activationFunction->derivative(this->values, upstreamGradients);
        this->biases -= learningRate * this->deltas;
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                this->weights[i][j] -= learningRate * nextLayer.deltas[j] * this->values[i];
            }
        }
    }
    void backward(const std::valarray<double>& actual, double learningRate) {
        assert(actual.size() == values.size());      //assertion
        this->deltas = activationFunction->derivative(this->values, (*lossFunction)(actual, this->values));
        this->biases -= learningRate * this->deltas;
    }
    ssize_t getLayerSize() const {
        return layerSize;
    }
    ssize_t getNextLayerSize() const {
        return nextLayerSize;
    }
    friend class Network;
    friend std::ostream& operator<< (std::ostream&, const Layer&);
    friend std::istream& operator>> (std::istream&, Layer&);
};

std::ostream& operator<< (std::ostream& os, const Layer& layer) {
    os << "<layer>" << "\r\n";
    os << "size: " << layer.getLayerSize() << "\r\n";
    os << "next-size: " << layer.getNextLayerSize() << "\r\n";
    os << "biases: ";
    for (ssize_t i = 0; i < layer.biases.size(); ++i) {
        os << layer.biases[i] << ' ';
    }
    os << "\r\n";
    os << "weights: ";
    for (ssize_t i = 0; i < layer.getLayerSize(); ++i) {
        for (ssize_t j = 0; j < layer.getNextLayerSize(); ++j) {
            os << layer.weights[i][j] << ' ';
        }
        os << "\r\n";
    }
    os << "activation-function: " << static_cast<size_t>(layer.activationFunctionEnum) << "\r\n";
    os << "loss-function: " << static_cast<size_t>(layer.lossFunctionEnum) << "\r\n";
    os << "</layer>";
    return os;
}

std::istream& operator>> (std::istream& is, Layer& layer) {
    static auto info_size = "size: "s;
    static auto info_next_size = "next-size: "s;
    static auto info_biases = "biases: "s;
    static auto info_weights = "weights: "s;
    static auto info_activation_function = "activation-function: "s;
    static auto info_loss_function = "loss-function: "s;
    std::string buffer;
    inputTill(is, buffer, "</layer>"s);
    std::string::const_iterator iter = std::search(buffer.cbegin(), buffer.cend(), info_size.cbegin(), info_size.cend());
    const char *ptr;
    std::advance(iter, info_size.length());
    ptr = &*iter;
    ssize_t size = std::atoll(ptr);

    iter = std::search(buffer.cbegin(), buffer.cend(), info_next_size.cbegin(), info_next_size.cend());
    std::advance(iter, info_next_size.length());
    ptr = &*iter;
    ssize_t next_size = std::atoll(ptr);

    iter = std::search(buffer.cbegin(), buffer.cend(), info_biases.cbegin(), info_biases.cend());
    std::advance(iter, info_biases.length());
    std::valarray<double> biases(size);
    ptr = &*iter;
    for (int i = 0; i < size; ++i) {
        auto [neo_ptr, ec] = std::from_chars(ptr, reinterpret_cast<const char *>(&*buffer.cend()), biases[i]);
        ptr = neo_ptr;
        ptr = std::find_if_not(ptr, &*buffer.cend(), [](char c){
            return std::isspace(c);
        });
    }

    iter = std::search(buffer.cbegin(), buffer.cend(), info_weights.cbegin(), info_weights.cend());
    std::advance(iter, info_weights.length());
    std::valarray<std::valarray<double>> weights(std::valarray<double>(next_size), size);
    ptr = &*iter;
    for (ssize_t i = 0; i < size; ++i) {
        for (ssize_t j = 0; j < next_size; ++j) {
            auto [neo_ptr, ec] = std::from_chars(ptr, reinterpret_cast<const char *>(&*buffer.cend()), weights[i][j]);
            ptr = neo_ptr;
            ptr = std::find_if_not(ptr, &*buffer.cend(), [](char c){
                return std::isspace(c);
            });
        }
    }

    iter = std::search(buffer.cbegin(), buffer.cend(), info_activation_function.cbegin(), info_activation_function.cend());
    std::advance(iter, info_activation_function.length());
    ptr = &*iter;
    ActivationFunctions activationFunctionEnum = static_cast<ActivationFunctions>(std::atoi(ptr));

    iter = std::search(buffer.cbegin(), buffer.cend(), info_loss_function.cbegin(), info_loss_function.cend());
    std::advance(iter, info_loss_function.length());
    ptr = &*iter;
    LossFunctions lossFunctionEnum = static_cast<LossFunctions>(std::atoi(ptr));

    Layer tmp(biases, weights, activationFunctionEnum, lossFunctionEnum);

    layer = std::move(tmp);

    return is;
}
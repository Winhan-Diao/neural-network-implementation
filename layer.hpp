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
    LossFunctions lossFunctionEnum;
    ssize_t layerSize;
    ssize_t nextLayerSize;
    std::unique_ptr<ActivationFunction> activationFunction;
    std::unique_ptr<LossFunction> lossFunction;
    std::valarray<double> momentumBiases;
    std::valarray<std::valarray<double>> momentumWeights;
    std::valarray<double> rmspropBiases;
    std::valarray<std::valarray<double>> rmspropWeights;
    static constexpr const double smoothingFactor = .01;
    static constexpr const double smallCorrection = .001;
    static constexpr const double decayFactor = .001;
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
        weights(std::valarray<double>(nextLayerNodeCounts), nodeCounts),     //bad perf 
        values(nodeCounts), 
        deltas(nodeCounts),
        activationFunctionEnum(activationFunctionEnum),
        lossFunctionEnum(lossFunctionEnum),
        layerSize(nodeCounts),
        nextLayerSize(weights.size()? weights[0].size(): 0),
        activationFunction(ActivationFunction::buildActivationFunction(activationFunctionEnum)), 
        lossFunction(LossFunction::buildLossFunction(lossFunctionEnum)),
        momentumBiases(nodeCounts),
        momentumWeights(std::valarray<double>(nextLayerNodeCounts), nodeCounts),
        rmspropBiases(nodeCounts),
        rmspropWeights(std::valarray<double>(nextLayerNodeCounts), nodeCounts)
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
        lossFunctionEnum(lossFunctionEnum),
        layerSize(weights.size()),
        nextLayerSize(weights.size()? weights[0].size(): 0),
        activationFunction(ActivationFunction::buildActivationFunction(activationFunctionEnum)),
        lossFunction(LossFunction::buildLossFunction(lossFunctionEnum)),
        momentumBiases(biases.size()),
        momentumWeights(std::valarray<double>(nextLayerSize), weights.size()),
        rmspropBiases(biases.size()),
        rmspropWeights(std::valarray<double>(nextLayerSize), weights.size())
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
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2.);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection);
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                momentumWeights[i][j] = (1 - smoothingFactor) * momentumWeights[i][j] + smoothingFactor * nextLayer.deltas[j] * this->values[i]; 
                rmspropWeights[i][j] = (1 - smoothingFactor) * rmspropWeights[i][j] + smoothingFactor * std::pow(nextLayer.deltas[j] * this->values[i], 2.); 
                this->weights[i][j] -= learningRate * momentumWeights[i][j] / (std::sqrt(rmspropWeights[i][j]) + smallCorrection) - learningRate * this->weights[i][j] * decayFactor;
            }
        }
    }
    void outputBackward(const std::valarray<double>& actual, double learningRate) {
        assert(actual.size() == values.size());      //assertion
        this->deltas = activationFunction->derivative(this->values, (*lossFunction)(actual, this->values));
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2.);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) - learningRate * this->biases * decayFactor;
    }
    void batchedBackward(const std::valarray<std::valarray<double>>& batchedValues, const Layer& nextLayer, double learningRate) {
        std::valarray<double> upstreamGradients(this->deltas.size());
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                upstreamGradients[i] += nextLayer.deltas[j] * this->weights[i][j];
            }
        }
        this->deltas = 0;
        for (ssize_t i = 0; i < batchedValues.size(); ++i) {
            this->deltas += activationFunction->derivative(batchedValues[i], upstreamGradients);
        }
        this->deltas /= batchedValues.size();

        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2.);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) - learningRate * this->biases * decayFactor;
        
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                momentumWeights[i][j] *= 1 - smoothingFactor;
                rmspropWeights[i][j] *= 1 - smoothingFactor;
                double deltaWeightGrad = 0;
                for (ssize_t h = 0; h < batchedValues.size(); ++h) {
                    deltaWeightGrad += nextLayer.deltas[j] * batchedValues[h][i]; 
                }
                momentumWeights[i][j] += smoothingFactor * deltaWeightGrad;
                rmspropWeights[i][j] += smoothingFactor * std::pow(deltaWeightGrad, 2.);
                this->weights[i][j] -= learningRate * momentumWeights[i][j] / (std::sqrt(rmspropWeights[i][j]) + smallCorrection) - learningRate * this->weights[i][j] * decayFactor;
            }
        }
    }
    void batchedOutputBackward(const std::valarray<std::valarray<double>>& batchedPredicted, const std::valarray<std::valarray<double>>& batchedActual, double learningRate) {
        assert(batchedPredicted.size() == batchedActual.size());
        this->deltas = 0;
        for (ssize_t i = 0; i < batchedPredicted.size(); ++i) {
            this->deltas += activationFunction->derivative(batchedPredicted[i], (*lossFunction)(batchedActual[i], batchedPredicted[i]));
        }
        this->deltas /= batchedPredicted.size();
        
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2.);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) - learningRate * this->biases * decayFactor;
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
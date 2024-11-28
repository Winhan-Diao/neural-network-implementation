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
#include "thread_pool.hpp"

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
    static constexpr const double smoothingFactor = 1.e-3;
    static constexpr const double smallCorrection = 1.e-10;
    static constexpr const double decayFactor = 1.e-8;
public:
    Layer(ssize_t nodeCounts
            , ssize_t nextLayerNodeCounts = 0
            , const ActivationFunctions& activationFunctionEnum = ActivationFunctions::SIGMOID
            , const LossFunctions& lossFunctionEnum = LossFunctions::MSE
            , std::mt19937 gen = std::mt19937(std::random_device()())
        ): 
        biases(nodeCounts), 
        weights(std::valarray<double>(nextLayerNodeCounts), nodeCounts),
        values(nodeCounts), 
        deltas(nodeCounts),
        activationFunctionEnum(activationFunctionEnum),
        lossFunctionEnum(lossFunctionEnum),
        layerSize(nodeCounts),
        nextLayerSize(weights.size()? weights[0].size(): 0),
        activationFunction(buildActivationFunction(activationFunctionEnum)), 
        lossFunction(buildLossFunction(lossFunctionEnum)),
        momentumBiases(nodeCounts),
        momentumWeights(std::valarray<double>(nextLayerNodeCounts), nodeCounts),
        rmspropBiases(nodeCounts),
        rmspropWeights(std::valarray<double>(nextLayerNodeCounts), nodeCounts)
    {
        for (ssize_t i = 0; i < biases.size(); ++i) {
            biases[i] = std::normal_distribution(0., .7)(gen);
        }
        for (ssize_t i = 0; i < weights.size(); ++i) {
            for (ssize_t j = 0; j < (weights.size()? weights[0].size(): 0); ++j) {
                if (activationFunctionEnum == ActivationFunctions::SIGMOID || activationFunctionEnum == ActivationFunctions::TANH)
                    weights[i][j] = std::normal_distribution(0., std::sqrt(2. / (nodeCounts + nextLayerNodeCounts)))(gen);
                else
                    weights[i][j] = std::normal_distribution(0., std::sqrt(2. / nodeCounts))(gen);
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
        activationFunction(buildActivationFunction(activationFunctionEnum)),
        lossFunction(buildLossFunction(lossFunctionEnum)),
        momentumBiases(biases.size()),
        momentumWeights(std::valarray<double>(nextLayerSize), weights.size()),
        rmspropBiases(biases.size()),
        rmspropWeights(std::valarray<double>(nextLayerSize), weights.size())
    {}
    Layer() = default;
    Layer(const Layer& l): 
        biases(l.biases),
        weights(l.weights),
        values(l.values),
        deltas(l.deltas),
        activationFunctionEnum(l.activationFunctionEnum),
        lossFunctionEnum(l.lossFunctionEnum),
        layerSize(l.layerSize),
        nextLayerSize(l.nextLayerSize),
        activationFunction(buildActivationFunction(activationFunctionEnum)),
        lossFunction(buildLossFunction(lossFunctionEnum)),
        momentumBiases(l.momentumBiases),
        momentumWeights(l.momentumWeights),
        rmspropBiases(l.rmspropBiases),
        rmspropWeights(l.rmspropWeights)
    {
        std::cout << "Layer Copy Constructor" << "\r\n";      //debug
    }
    Layer& operator=(const Layer& l) {
        biases = l.biases;
        weights = l.weights;
        values = l.values;
        deltas = l.deltas;
        activationFunctionEnum = l.activationFunctionEnum;
        lossFunctionEnum = l.lossFunctionEnum;
        layerSize = l.layerSize;
        nextLayerSize = l.nextLayerSize;
        momentumBiases = l.momentumBiases;
        momentumWeights = l.momentumWeights;
        rmspropBiases = l.rmspropBiases;
        rmspropWeights = l.rmspropWeights;
        activationFunction = buildActivationFunction(activationFunctionEnum);
        lossFunction = buildLossFunction(lossFunctionEnum);
        std::cout << "Layer Copy Assignment" << "\r\n";      //debug
        return *this;
    }
    void forward(const Layer& prevLayer) {
        std::valarray<double> tmpValarr(this->values.size());
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < prevLayer.weights.size(); ++j) {
                tmpValarr[i] += prevLayer.weights[j][i] * prevLayer.values[j];
            }
        }
        this->values = (*activationFunction)(static_cast<std::valarray<double>&&>(this->biases + std::move(tmpValarr)));
    }
    std::valarray<double> externForward(const Layer& prevLayer, const std::valarray<double>& prevValues) const {
        std::valarray<double> tmpValarr(this->values.size());
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < prevLayer.weights.size(); ++j) {
                tmpValarr[i] += prevLayer.weights[j][i] * prevValues[j];
            }
        }
        return (*activationFunction)(static_cast<std::valarray<double>&&>(this->biases + std::move(tmpValarr)));
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
                rmspropWeights[i][j] = (1 - smoothingFactor) * rmspropWeights[i][j] + smoothingFactor * std::pow(nextLayer.deltas[j] * this->values[i], 2); 
                this->weights[i][j] -= learningRate * momentumWeights[i][j] / (std::sqrt(rmspropWeights[i][j]) + smallCorrection) + learningRate * this->weights[i][j] * decayFactor;
            }
        }
    }
    void outputBackward(const std::valarray<double>& actual, double learningRate) {
        assert(actual.size() == values.size());      //assertion
        this->deltas = activationFunction->derivative(this->values, (*lossFunction)(actual, this->values));
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) + learningRate * this->biases * decayFactor;
    }
    void outputBackward(double actual, size_t index, double learningRate) {
        assert(index < values.size());      //assertion
        std::fill(std::begin(this->deltas), std::end(this->deltas), double(0));
        double loss = (*lossFunction)(actual, this->values[index]);
        std::valarray<double> losses(double(0), values.size());
        losses[index] = loss;
        this->deltas = activationFunction->derivative(this->values, losses);
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) + learningRate * this->biases * decayFactor;
    }
    std::valarray<std::valarray<double>> batchedBackward(const std::valarray<std::valarray<double>>& batchedValues, const std::valarray<std::valarray<double>>& batchedNextDeltas, const Layer& nextLayer, double learningRate, size_t threadCounts = 1) {
        std::valarray<std::valarray<double>> batchedUpstreamGradients(std::valarray<double>(0.0, this->deltas.size()), batchedValues.size());
        std::valarray<std::valarray<double>> batchedDeltas(std::valarray<double>(this->deltas.size()), batchedValues.size());
        if (threadCounts > 1) {
            ThreadPool threadPool(threadCounts);
            for (ssize_t h = 0; h < batchedValues.size(); ++h) {
                threadPool.addTasks([](size_t b
                                        , std::valarray<double>& upstreamGradients
                                        , const std::valarray<double>& nextDeltas
                                        , const std::valarray<std::valarray<double>>& thisWeights
                                        , std::valarray<double>& thisDeltas
                                        , const std::valarray<double>& thisValues
                                        , ActivationFunctions activationFunctionEnum
                                    ) {
                    for (ssize_t i = 0; i < thisValues.size(); ++i) {
                        for (ssize_t j = 0; j < nextDeltas.size(); ++j) {
                            upstreamGradients[i] += nextDeltas[j] * thisWeights[i][j];
                        }
                    }
                    thisDeltas = buildActivationFunction(activationFunctionEnum)->derivative(thisValues, upstreamGradients);
                }, h, std::ref(batchedUpstreamGradients[h]), std::cref(batchedNextDeltas[h]), std::cref(this->weights), std::ref(batchedDeltas[h]), std::cref(batchedValues[h]), activationFunctionEnum);
            }
        } else {
            for (ssize_t h = 0; h < batchedValues.size(); ++h) {
                for (ssize_t i = 0; i < this->values.size(); ++i) {
                    for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                        batchedUpstreamGradients[h][i] += batchedNextDeltas[h][j] * this->weights[i][j];
                    }
                }
                batchedDeltas[h] = activationFunction->derivative(batchedValues[h], batchedUpstreamGradients[h]);
            }
        }
        this->deltas = batchedDeltas.sum() / batchedDeltas.size();

        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) + learningRate * this->biases * decayFactor;
        
        for (ssize_t i = 0; i < this->values.size(); ++i) {
            for (ssize_t j = 0; j < nextLayer.values.size(); ++j) {
                momentumWeights[i][j] *= 1 - smoothingFactor;
                rmspropWeights[i][j] *= 1 - smoothingFactor;
                double deltaWeightGrad = 0;
                for (ssize_t h = 0; h < batchedValues.size(); ++h) {
                    deltaWeightGrad += batchedNextDeltas[h][j] * batchedValues[h][i]; 
                }
                deltaWeightGrad /= batchedValues.size();
                momentumWeights[i][j] += smoothingFactor * deltaWeightGrad;
                rmspropWeights[i][j] += smoothingFactor * std::pow(deltaWeightGrad, 2);
                this->weights[i][j] -= learningRate * momentumWeights[i][j] / (std::sqrt(rmspropWeights[i][j]) + smallCorrection) + learningRate * this->weights[i][j] * decayFactor;
            }
        }
        return batchedDeltas;
    }
    std::valarray<std::valarray<double>> batchedOutputBackward(const std::valarray<std::valarray<double>>& batchedPredicted, const std::valarray<std::valarray<double>>& batchedActual, double learningRate, size_t threadCounts = 1) {
        assert(batchedPredicted.size() == batchedActual.size());
        std::valarray<std::valarray<double>> batchedDeltas(std::valarray<double>(this->deltas.size()), batchedPredicted.size());
        // this->deltas = 0;
        if (threadCounts > 1) {
            ThreadPool threadPool(threadCounts);
            for (size_t i = 0; i < batchedPredicted.size(); ++i)
                threadPool.addTasks([](size_t b
                                        , std::valarray<double>& thisDeltas
                                        , const std::valarray<double>& predicted
                                        , const std::valarray<double>& actual
                                        , ActivationFunctions activationFunction
                                        , LossFunctions lossFunctionEnum
                                    ) {
                    thisDeltas = buildActivationFunction(activationFunction)->derivative(predicted, (*buildLossFunction(lossFunctionEnum))(actual, predicted));
                }, i, std::ref(batchedDeltas[i]), std::cref(batchedPredicted[i]), std::cref(batchedActual[i]), activationFunctionEnum, lossFunctionEnum);
        } else {
            for (size_t i = 0; i < batchedPredicted.size(); ++i) {
                batchedDeltas[i] = activationFunction->derivative(batchedPredicted[i], (*lossFunction)(batchedActual[i], batchedPredicted[i]));
            }
        }
        this->deltas = batchedDeltas.sum() / batchedDeltas.size();
        
        momentumBiases = (1 - smoothingFactor) * momentumBiases + smoothingFactor * this->deltas;
        rmspropBiases = (1 - smoothingFactor) * rmspropBiases + smoothingFactor * std::pow(this->deltas, 2);
        this->biases -= learningRate * this->momentumBiases / (std::sqrt(rmspropBiases) + smallCorrection) + learningRate * this->biases * decayFactor;
        return batchedDeltas;
    }
    ssize_t getLayerSize() const {
        return layerSize;
    }
    ssize_t getNextLayerSize() const {
        return nextLayerSize;
    }
    // std::valarray<double>& getValues() {}
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
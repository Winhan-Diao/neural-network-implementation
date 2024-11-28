#pragma once
#include <iostream>
#include <valarray>
#include <vector>
#include <type_traits>
#include <cassert>
#include <algorithm>
#include <random>
#include <string>
#include <functional>
#include "layer.hpp"
#include "traits.hpp"
#include "stream_utils.hpp"
#include "thread_pool.hpp"

using namespace std::literals;

class Network {
    Layer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;
public:
    template <class I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Network(ssize_t inputLayerNodeCounts
                , ssize_t outputLayerNodeCounts
                , const std::vector<I>& hiddenLayersNodeCounts
                , const std::vector<ActivationFunctions>& hiddenLayersActivationFunctionEnum = {}
                , const ActivationFunctions& outputLayerActivationFunctionEnum = ActivationFunctions::SOFTMAX
                , const LossFunctions& outputLayerLossFunctionEnum = LossFunctions::CROSS_ENTROPY_LOSS
                , bool balanced = false
            ): 
        inputLayer(inputLayerNodeCounts
                    , (hiddenLayersNodeCounts.size() == 0)? outputLayerNodeCounts: hiddenLayersNodeCounts.at(0)
                ),
        hiddenLayers(),
        outputLayer(outputLayerNodeCounts
                        , 0
                        , outputLayerActivationFunctionEnum
                        , outputLayerLossFunctionEnum
                    )
    {
        assert(hiddenLayersActivationFunctionEnum.size() == hiddenLayersNodeCounts.size() || !hiddenLayersActivationFunctionEnum.size());
        for (ssize_t i = 0; i < hiddenLayersNodeCounts.size(); ++i) {
            hiddenLayers.emplace_back(hiddenLayersNodeCounts.at(i)
                                        , (i + 1 < hiddenLayersNodeCounts.size())? hiddenLayersNodeCounts.at(i): outputLayerNodeCounts
                                        , hiddenLayersActivationFunctionEnum.size()? hiddenLayersActivationFunctionEnum[i]: ActivationFunctions::LEAKYRELU
                                    );
        }
        if (balanced) {
            for (std::valarray<double>& weight: hiddenLayers.back().weights)
                std::fill(std::begin(weight) + 1, std::end(weight), weight[0]);
            std::fill(std::begin(outputLayer.biases) + 1, std::end(outputLayer.biases), outputLayer.biases[0]);
        }
    }
    template <
        class T, 
        class... _Arg,
        template <class, class> class V, 
        typename = std::enable_if_t <
            std::is_same_v<std::decay_t<T>, Layer> 
            && std::is_constructible_v<std::vector<Layer, _Arg...>, V<T, _Arg...>>
        >
    >
    Network(T&& inputLayer, V<T, _Arg...>&& hiddenLayers, T&& outputLayer)
        : inputLayer(std::forward<T>(inputLayer)) 
        , hiddenLayers(std::forward<V<T, _Arg...>>(hiddenLayers))
        , outputLayer(std::forward<T>(outputLayer))
    {}
    Network() = default;
    std::valarray<double> train(const std::valarray<double>& input, const std::valarray<double>& output, double learningRate) {
        assert(input.size() == inputLayer.values.size());       //assertion
        assert(output.size() == outputLayer.values.size());       //assertion
        run(input);
        outputLayer.outputBackward(output, learningRate);
        for (ssize_t i = hiddenLayers.size() - 1; i >= 0; --i) {
            hiddenLayers[i].backward((i == hiddenLayers.size() - 1)? outputLayer: hiddenLayers[i + 1], learningRate);
        }
        inputLayer.backward(hiddenLayers[0], learningRate);
        return outputLayer.values;
    }
    std::valarray<double> train(const std::valarray<double>& input, double output, size_t index, double learningRate) {
        assert(input.size() == inputLayer.values.size());       //assertion
        assert(index < outputLayer.values.size());       //assertion
        run(input);
        outputLayer.outputBackward(output, index, learningRate);
        for (ssize_t i = hiddenLayers.size() - 1; i >= 0; --i) {
            hiddenLayers[i].backward((i == hiddenLayers.size() - 1)? outputLayer: hiddenLayers[i + 1], learningRate);
        }
        inputLayer.backward(hiddenLayers[0], learningRate);
        return outputLayer.values;
    }
    void batchedTrain(const std::valarray<std::valarray<double>>& batchedInput, const std::valarray<std::valarray<double>>& batchedOutput, double learningRate, size_t threadCounts = 1) {
        assert(batchedInput.size() == batchedOutput.size());       //assertion
        // z: Layers; y: batches; x: nodes
        std::vector<std::valarray<std::valarray<double>>> batchedHiddenLayersValues(size_t(hiddenLayers.size()), std::valarray<std::valarray<double>>(size_t(batchedInput.size())));
        // y: batches; x: nodes
        std::valarray<std::valarray<double>> batchedOutputLayerValues(size_t(batchedInput.size()));
        if (threadCounts > 1) {
            ThreadPool threadPool(threadCounts);
            for (size_t i = 0; i < batchedInput.size(); ++i) {
                threadPool.addTasks([](size_t b
                                        , const std::valarray<double>& inputValues
                                        , std::vector<std::valarray<std::valarray<double>>>& batchedHiddenLayersValues
                                        , std::valarray<double>& outputLayerValues
                                        , const Layer& inputLayer
                                        , const std::vector<Layer>& hiddenLayers
                                        , const Layer& outputLayer
                                    ) {
                    for (ssize_t j = 0; j < batchedHiddenLayersValues.size(); ++j) {
                        batchedHiddenLayersValues[j][b] = hiddenLayers[j].externForward(j? hiddenLayers[j - 1]: inputLayer, j? batchedHiddenLayersValues[j - 1][b]: inputValues);
                    }
                    outputLayerValues = outputLayer.externForward(hiddenLayers.back(), batchedHiddenLayersValues.back()[b]);
                }, i, std::cref(batchedInput[i]), std::ref(batchedHiddenLayersValues), std::ref(batchedOutputLayerValues[i]), std::cref(inputLayer), std::cref(hiddenLayers), std::cref(outputLayer));
            }
        } else {
            for (size_t i = 0; i < batchedInput.size(); ++i) {
                for (ssize_t j = 0; j < batchedHiddenLayersValues.size(); ++j) {
                    batchedHiddenLayersValues[j][i] = hiddenLayers[j].externForward(j? hiddenLayers[j - 1]: inputLayer, j? batchedHiddenLayersValues[j - 1][i]: batchedInput[i]);
                }
                batchedOutputLayerValues[i] = outputLayer.externForward(hiddenLayers.back(), batchedHiddenLayersValues.back()[i]);
            }
        }

        std::valarray<std::valarray<double>> batchedDeltas = outputLayer.batchedOutputBackward(batchedOutputLayerValues, batchedOutput, learningRate, threadCounts);
        for (ssize_t i = hiddenLayers.size() - 1; i >= 0; --i) {
            batchedDeltas = hiddenLayers[i].batchedBackward(batchedHiddenLayersValues[i], batchedDeltas, (i == hiddenLayers.size() - 1)? outputLayer: hiddenLayers[i + 1], learningRate, threadCounts);
        }
        inputLayer.batchedBackward(batchedInput, batchedDeltas, hiddenLayers[0], learningRate, threadCounts);
        return;
    }
    std::valarray<double> run(const std::valarray<double>& input) {
        inputLayer.values = input;
        for (ssize_t i = 0; i < hiddenLayers.size(); ++i) {
            hiddenLayers[i].forward((i == 0)? inputLayer: hiddenLayers.at(i - 1));
        }
        outputLayer.forward(hiddenLayers.back());
        return outputLayer.values;
    }
    template <class _Actual, class _BiPred>
    bool test(const std::valarray<double>& testInputs, _Actual&& testActual, _BiPred&& biPred) {
        std::valarray<double> testPredicted = this->run(testInputs);
        return biPred(testPredicted, testActual);
    }
    template <class _Actual, class _BiPred>
    double test(const std::valarray<std::valarray<double>>& testInputs, _Actual&& testActual, _BiPred&& biPred) {
        ssize_t correctCounts = 0;
        for (ssize_t i = 0; i < testInputs.size(); ++i) {
            if (biPred(this->run(testInputs[i]), testActual[i]))
                ++correctCounts;
        }
        return correctCounts / static_cast<double>(testInputs.size());
    }
    void assignData(const Network& n) {
        inputLayer.weights = n.inputLayer.weights;
        inputLayer.biases = n.inputLayer.biases;
        for (size_t i = 0; i < hiddenLayers.size(); ++i) {
            hiddenLayers[i].weights = n.hiddenLayers[i].weights;
            hiddenLayers[i].biases = n.hiddenLayers[i].biases;
        }
        outputLayer.weights = n.outputLayer.weights;
        outputLayer.biases = n.outputLayer.biases;
    }
    friend inline std::ostream& operator<< (std::ostream&, const Network&);
    friend inline std::istream& operator>> (std::istream&, Network&);
};

inline std::ostream& operator<< (std::ostream& os, const Network& network) {
    os << "<network>" << "\r\n";
    os << "<hidden-layers-counts>" << "\r\n";
    os << network.hiddenLayers.size() << "\r\n";
    os << "</hidden-layers-counts>" << "\r\n";
    os << "<input-layer>" << "\r\n";
    os << network.inputLayer;
    os << "</input-layer>" << "\r\n";
    os << "<hidden-layers>" << "\r\n";
    for (ssize_t i = 0; i < network.hiddenLayers.size(); ++i) {
        os << "<" << i << ">" << "\r\n";
        os << network.hiddenLayers[i] << "\r\n";
        os << "</" << i << ">" << "\r\n";
    }
    os << "</hidden-layers>" << "\r\n";
    os << "<output-layer>" << "\r\n";
    os << network.outputLayer << "\r\n";
    os << "</output-layer>" << "\r\n";
    os << "</network>";
    return os;
}

inline std::istream& operator>> (std::istream& is, Network& network) {
    skipTill(is, "<hidden-layers-counts>"s);
    ssize_t hidden_size;
    is >> hidden_size;
    
    skipTill(is, "<input-layer>"s);
    Layer inputLayer;
    is >> inputLayer;

    skipTill(is, "<hidden-layers>"s);
    std::vector<Layer> hiddenLayers(hidden_size);
    for (ssize_t i = 0; i < hidden_size; ++i) {
        is >> hiddenLayers[i];
    }

    skipTill(is, "<output-layer>"s);
    Layer outputLayer;
    is >> outputLayer;

    skipTill(is, "</network>"s);
    Network tmp(std::move(inputLayer), std::move(hiddenLayers), std::move(outputLayer));
    network = std::move(tmp);
    return is;
}
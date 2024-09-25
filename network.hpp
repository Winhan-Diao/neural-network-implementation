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

using namespace std::literals;

template <template <typename> typename T, typename V,  class U = decltype("valarr"s)>
void printValarray(const T<V>& valarr, U&& name = "valarr"s) {
    std::cout << name << ": {"s;
    std::for_each(std::cbegin(valarr), std::cend(valarr), [&valarr](const auto& a){
        if constexpr (is_valarray<V>::value) {
            printValarray(a, "innerVArr"s);
        } else {
            std::cout << a;
        }
        std::cout << ((&a == std::cend(valarr) - 1)? ""s : ", "s);
    });
    std::cout << "}"s << "\r\n"s;
}

class Network {
    Layer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;
public:
    template <class I, typename = std::enable_if_t<std::is_integral_v<I>>>
    Network(ssize_t inputLayerNodeCounts, ssize_t outputLayerNodeCounts, const std::vector<I>& hiddenLayersNodeCounts): 
        inputLayer(inputLayerNodeCounts
                    , (hiddenLayersNodeCounts.size() == 0)? outputLayerNodeCounts: hiddenLayersNodeCounts.at(0)
                    , std::mt19937(std::random_device()())
                    , std::normal_distribution(0., sqrt(2. / (inputLayerNodeCounts + outputLayerNodeCounts)))
                    , std::normal_distribution(0., .001)
                ),
        hiddenLayers(),
        outputLayer(outputLayerNodeCounts
                        , 0
                        , std::mt19937(std::random_device()())
                        , std::normal_distribution(0., sqrt(2. / (inputLayerNodeCounts + outputLayerNodeCounts)))
                        , std::normal_distribution(0., .001)
                        , ActivationFunctions::SOFTMAX
                        , LossFunctions::CROSS_ENTROPY_LOSS
                    )
    {
        for (ssize_t i = 0; i < hiddenLayersNodeCounts.size(); ++i) {
            hiddenLayers.emplace_back(hiddenLayersNodeCounts.at(i)
                                        , (i + 1 < hiddenLayersNodeCounts.size())? hiddenLayersNodeCounts.at(i): outputLayerNodeCounts
                                        , std::mt19937(std::random_device()())
                                        , std::normal_distribution(0., sqrt(2. / (inputLayerNodeCounts + outputLayerNodeCounts)))
                                        , std::normal_distribution(0., .001)
                                        , ActivationFunctions::LEAKYRELU
                                    );
        }
    }
    template <
        class T, 
        class _Ig,
        template <class, class> class V, 
        typename = std::enable_if_t <
            std::is_same_v<std::decay_t<T>, Layer> 
            && std::is_constructible_v<std::vector<Layer, _Ig>, V<T, _Ig>>
        >
    >
    Network(T&& inputLayer, V<T, _Ig>&& hiddenLayers, T&& outputLayer)
        : inputLayer(std::forward<T>(inputLayer)) 
        , hiddenLayers(std::forward<V<T, _Ig>>(hiddenLayers))
        , outputLayer(std::forward<T>(outputLayer))
    {}
    Network() = default;
    std::valarray<double> train(const std::valarray<double>& input, const std::valarray<double>& output, double learningRate) {
        assert(input.size() == inputLayer.values.size());       //assertion
        assert(output.size() == outputLayer.values.size());       //assertion
        inputLayer.values = input;
        for (ssize_t i = 0; i < hiddenLayers.size(); ++i) {
            hiddenLayers[i].forward((i == 0)? inputLayer: hiddenLayers.at(i - 1));
        }
        outputLayer.forward(hiddenLayers.back());
        outputLayer.backward(output, learningRate);
        for (ssize_t i = hiddenLayers.size() - 1; i >= 0; --i) {
            hiddenLayers[i].backward((i == hiddenLayers.size() - 1)? outputLayer: hiddenLayers[i + 1], learningRate);
        }
        inputLayer.backward(hiddenLayers[0], learningRate);
        return outputLayer.values;
    }
    std::valarray<double> run(const std::valarray<double>& input, const std::valarray<double>& output) {
        return this->train(input, output, 0);
    }
    friend int main();
    friend std::ostream& operator<< (std::ostream&, const Network&);
    friend std::istream& operator>> (std::istream&, Network&);
};

std::ostream& operator<< (std::ostream& os, const Network& network) {
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

std::istream& operator>> (std::istream& is, Network& network) {
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
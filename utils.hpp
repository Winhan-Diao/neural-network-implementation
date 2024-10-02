#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <valarray>
#include <cassert>
#include "network.hpp"

std::vector<size_t> generateShuffledIndices(size_t size) {
    std::vector<size_t> indices(size_t(size), size_t(0));
    std::iota(std::min(indices.begin() + 1, indices.end()), indices.end(), 1);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
    return indices;
}

template <class T>
std::valarray<T> reorder(const std::valarray<T>& src, const std::vector<size_t>& indices) {
    assert(src.size() == indices.size());       //assertion
    std::valarray<T> neo(src.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        neo[i] = src[indices[i]];
    }
    return neo;
}

template <class T>
std::valarray<std::valarray<T>> batch(const std::valarray<T>& origin, size_t batchSize, const std::vector<size_t>& indices) {
    assert(origin.size() == indices.size());       //assertion
    std::valarray<T> shuffled = reorder(origin, indices);
    size_t size = shuffled.size() / batchSize;
    std::valarray<std::valarray<T>> batched(std::valarray<T>(batchSize), size);
    for (size_t i = 0; i < size; ++i) {
        batched[i] = shuffled[std::slice(i * batchSize, batchSize, 1)];
    }
    return batched;
}

template <class T>
std::valarray<std::valarray<T>> batch(const std::valarray<T>& origin, size_t batchSize) {
    size_t size = origin.size() / batchSize;
    std::valarray<std::valarray<T>> batched(std::valarray<T>(batchSize), size);
    for (size_t i = 0; i < size; ++i) {
        batched[i] = origin[std::slice(i * batchSize, batchSize, 1)];
    }
    return batched;
}

template <class T1, class T2, class _BiPred>
void train(Network& n, const std::valarray<std::valarray<double>>& trainInputs, const std::valarray<T1>& trainOutputs, double learningRate, size_t epoch, size_t batchSize, const std::valarray<std::valarray<double>>& testInputs, const std::valarray<T2>& testOutputs, _BiPred&& testBiPred) {
    assert(trainInputs.size() == trainOutputs.size());       //assertion
    for (size_t e = 0; e < epoch; ++e) {
        std::cout << "epoch " << e << "\r\n";
        std::vector<size_t> indices = generateShuffledIndices(trainInputs.size());
        auto shuffledInputs = reorder(trainInputs, indices);
        auto shuffledOutputs = reorder(trainOutputs, indices);
        auto batchedInputs = batch(shuffledInputs, batchSize);
        auto batchedOutputs = batch(shuffledOutputs, batchSize);
        size_t p = 0;
        std::cout << "training";
        for (size_t b = 0; b < batchedInputs.size(); ++b) {
            n.batchedTrain(batchedInputs[b], batchedOutputs[b], learningRate * batchSize);
            if (b * batchSize > p) {
                std::cout << '.';
                p += indices.size() / 20;
            }
        }
        std::cout << "\r\n" << "all batched data is trained" << "\r\n";
        std::cout << "assessing accuracy: " << n.test(testInputs, testOutputs, testBiPred) << "\r\n";
    }
}
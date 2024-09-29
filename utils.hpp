#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <valarray>
#include <cassert>

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


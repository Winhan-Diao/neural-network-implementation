#pragma once
#include <fstream>
#include <valarray>
#include <iostream>

using namespace std::string_literals;

std::valarray<double> loadLabels(const std::string& loc) {
    if (std::ifstream ifs{loc, std::ios::binary}) {
        unsigned char tmp;
        unsigned ignoredU;
        int counts;
        ifs.read(reinterpret_cast<char *>(&ignoredU), sizeof(ignoredU));
        ifs.read(reinterpret_cast<char *>(&counts), sizeof(counts));
        std::reverse(reinterpret_cast<char *>(&counts), reinterpret_cast<char *>(&counts) + sizeof(counts));
        std::valarray<double> buffer(counts);
        for (int i = 0; i < counts; ++i) {
            ifs.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
            buffer[i] = tmp;
        }
        return buffer;
    }
    throw std::runtime_error{"can't open "s + loc + " to load labels"s};
}

int getCounts(const std::string& loc) {
    if (std::ifstream ifs{loc, std::ios::binary}) {
        int ignored;
        int counts;
        ifs.read(reinterpret_cast<char *>(&ignored), sizeof(ignored));
        ifs.read(reinterpret_cast<char *>(&counts), sizeof(counts));
        std::reverse(reinterpret_cast<char *>(&counts), reinterpret_cast<char *>(&counts) + sizeof(counts));
        return counts;
    }
    throw std::runtime_error{"can't open "s + loc + " to get labels or images counts"s};
}

std::valarray<std::valarray<double>> loadImages(const std::string& loc) {
    if (std::ifstream ifs{loc, std::ios::binary}) {
        unsigned char tmp;
        unsigned ignoredU;
        int counts;
        ifs.read(reinterpret_cast<char *>(&ignoredU), sizeof(ignoredU));
        ifs.read(reinterpret_cast<char *>(&counts), sizeof(counts));
        std::reverse(reinterpret_cast<char *>(&counts), reinterpret_cast<char *>(&counts) + sizeof(counts));
        ifs.read(reinterpret_cast<char *>(&ignoredU), sizeof(ignoredU));
        ifs.read(reinterpret_cast<char *>(&ignoredU), sizeof(ignoredU));
        std::valarray<std::valarray<double>> buffer(std::valarray<double>(28*28), counts);
        for (int i = 0; i < counts; ++i) {
            for (int j = 0; j < 28*28; ++j) {
                ifs.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
                buffer[i][j] = tmp;
            }
        }
        return buffer;
    }
    throw std::runtime_error{"can't open "s + loc + " to load images"s};
}

std::valarray<std::valarray<double>> classifyLabels(const std::valarray<double>& orignal) {
    std::valarray<std::valarray<double>> neo(std::valarray<double>(10), orignal.size());
    for (int i = 0; i < orignal.size(); ++i) {
        neo[i][orignal[i]] = 1;
    }
    return neo;
}

std::ptrdiff_t getGreatestLabel(const std::valarray<double>& labels) {
    return std::distance(std::cbegin(labels), std::max_element(std::cbegin(labels), std::cend(labels)));
}

void printImage(const std::valarray<double>& image) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (image[i * 28 + j] > 0? "▓"s: "░"s);
        }
        std::cout << "\r\n";
    }
}

// template <class RandomGenerator>
// std::valarray<double> randomOffset(std::valarray<double>& image, RandomGenerator&& gen, double stdDerivation = 1) {
//     int offsetX = std::normal_distribution(0, stdDerivation)(gen);
//     int offsetY = std::normal_distribution(0, stdDerivation)(gen);
//     if (offsetX < 0) {
//         if (std::all_of(std::cbegin(image), std::cbegin(image) + offsetX * 28, [](const auto& v){ return v == 0; }))
        
//     } else if (offsetX > 0) {

//     }
//     if (offsetY < 0) {

//     } else if (offsetY > 0) {

//     }
// }
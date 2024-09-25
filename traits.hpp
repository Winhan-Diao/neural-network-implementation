#pragma once
#include <valarray>

template <class T>
struct is_valarray: std::false_type {};

template <class T>
struct is_valarray<std::valarray<T>>: std::true_type {};

template <class T>
static constexpr bool is_valarray_v = is_valarray<T>::value;

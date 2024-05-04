#pragma once

#include <Eigen/Dense>

template <typename T=float>
requires std::is_arithmetic_v<T>
using ImArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T=float>
requires std::is_arithmetic_v<T>
using ImArrayRef = Eigen::Ref<ImArray<T>>;

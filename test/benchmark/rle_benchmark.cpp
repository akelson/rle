#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <iostream>
#include "codec/rle_v1.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

static void BM_encode_v1(benchmark::State &state)
{
    Array<float, Dynamic, Dynamic> rand_x = Matrix<float, Dynamic, Dynamic>::Random(648, 480);
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x = (rand_x > 0.9).cast<uint8_t>();

    std::vector<uint8_t> buff(10e6);

    std::span<uint8_t> rle;
    for (auto _ : state)
        rle = rle::v1::encode(std::span(x.data(), x.size()), buff);
}
BENCHMARK(BM_encode_v1)->Unit(benchmark::kMillisecond);
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include "rle.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

static void BM_encode(benchmark::State &state)
{
    Array<float, Dynamic, Dynamic> rand_x = Matrix<float, Dynamic, Dynamic>::Random(640, 480);
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x = (rand_x > 1.0).cast<uint8_t>();

    x.block(500, 500, 200, 100) = 1;
    x.block(5, 5, 2, 2) = 1;

    std::vector<uint8_t> buff(10e6);

    for (auto _ : state)
        std::span<uint8_t> rle = encode(std::span(x.data(), x.size()), buff);
}
BENCHMARK(BM_encode)->Unit(benchmark::kMillisecond);
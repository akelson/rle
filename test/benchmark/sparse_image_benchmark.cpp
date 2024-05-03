
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <iostream>
#include "imgproc/sparse_image.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

static void BM_sparse_image_ctor(benchmark::State &state)
{
    ImArray<float> rand_x = Matrix<float, Dynamic, Dynamic>::Random(648, 480);
    const float threshold = 0.01 * state.range(0);
    ImArray<bool> x = rand_x > threshold;

    std::vector<uint8_t> buff(10e6);

    for (auto _ : state)
    {
        SparseImage sparse_image(x);
    }
}
BENCHMARK(BM_sparse_image_ctor)
    ->DenseRange(80, 100, 2)
    ->Unit(benchmark::kMillisecond);

static void BM_from_sparse_image(benchmark::State &state)
{
    ImArray<float> rand_x = Matrix<float, Dynamic, Dynamic>::Random(648, 480);
    const float threshold = 0.01 * state.range(0);
    ImArray<bool> x = rand_x > threshold;

    std::vector<uint8_t> buff(10e6);
    SparseImage sparse_image(x);

    ImArray<bool> x_out;
    x_out.resizeLike(x);

    for (auto _ : state)
    {
        from_sparse_image<bool>(sparse_image, x_out);
    }
}
BENCHMARK(BM_from_sparse_image)
    ->DenseRange(80, 100, 2)
    ->Unit(benchmark::kMillisecond);

static void BM_copy_image_dense(benchmark::State &state)
{
    ImArray<float> rand_x = Matrix<float, Dynamic, Dynamic>::Random(648, 480);
    ImArray<bool> x = rand_x > 0.95;

    ImArray<bool> x_out;
    x_out.resizeLike(x);

    for (auto _ : state)
    {
        x_out = x;
    }
}
BENCHMARK(BM_copy_image_dense)->Unit(benchmark::kMillisecond);
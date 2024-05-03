#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "sparse_image.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using ::testing::ElementsAre;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

TEST(SparseImage, test_1)
{
    Array<bool, Dynamic, Dynamic, RowMajor> x(2, 3);
    x.setConstant(0);
    x(0, 0) = true;
    x(1, 1) = true;

    SparseImage<bool> sparse_image(x);

    EXPECT_EQ(sparse_image.width(), x.cols());
    EXPECT_EQ(sparse_image.height(), x.rows());
    EXPECT_EQ(sparse_image.size(), x.rows() * x.cols());

    std::array<bool, 6> out;
    for(auto it = sparse_image.begin(); it != sparse_image.end(); ++it)
    {
        bool val = *it;
        out[it.index()] = val;
    }
    ImArray<bool> x_out = Map<Array<bool, 2, 3, RowMajor>>(out.data());
    EXPECT_TRUE((x_out == x).all());
}

TEST(SparseImage, from_sparse_image)
{
    ImArray<bool> x(3, 4);
    x.setConstant(0);
    x(0, 0) = true;
    x(1, 2) = true;

    ImArray<bool> x_out(3, 4);

    const SparseImage<bool> sparse_image(x);

    from_sparse_image<bool>(sparse_image, x_out);

    EXPECT_TRUE((x_out == x).all());
}

TEST(SparseImage, correlate)
{
    ImArray<bool> x(3, 4);
    x.setConstant(0);
    x(1, 1) = 1;

    SparseImage<bool> sparse_image(x);

    ImArray<uint8_t> kernel
    {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };

    ImArray<uint8_t> out(3, 4);
    out.setConstant(0);

    correlate<uint8_t>(sparse_image, kernel, out);

    ImArray<uint8_t> out_expected
    {
        {1, 2, 3, 0},
        {4, 5, 6, 0},
        {7, 8, 9, 0}
    };

    EXPECT_TRUE((out == out_expected).all());

    DISP(out);
}
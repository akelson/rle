#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "imgproc/sparse_image.h"
#include "imgproc/cwise_binary_op.h"
#include "imgproc/ops.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using ::testing::ElementsAre;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

TEST(SparseImage, ctor_all_zeros)
{
    ImArray<bool> x(2, 3);
    x.setConstant(0);

    SparseImage<bool> sparse_image(x);

    EXPECT_EQ(sparse_image.width(), x.cols());
    EXPECT_EQ(sparse_image.height(), x.rows());
    EXPECT_EQ(sparse_image.size(), x.rows() * x.cols());
};

TEST(SparseImage, iterate_all_zeros)
{
    ImArray<bool> x(2, 3);
    x.setConstant(0);

    SparseImage<bool> sparse_image(x);

    EXPECT_EQ(sparse_image.begin(), sparse_image.end());
}

TEST(SparseImage, all_zeros_large)
{
    ImArray<bool> x(4000, 3000);
    x.setConstant(0);

    SparseImage<bool> sparse_image(x);

    EXPECT_EQ(sparse_image.begin(), sparse_image.end());
}

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

    std::array<bool, 6> out{};
    for(auto it = sparse_image.begin(); it != sparse_image.end(); ++it)
    {
        bool val = *it;
        out[it.index()] = val;
    }
    ImArray<bool> x_out = Map<Array<bool, 2, 3, RowMajor>>(out.data());
    EXPECT_TRUE((x_out == x).all())
        << x_out;
}

TEST(SparseImage, from_sparse_image)
{
    ImArray<bool> x(3, 4);
    x.setConstant(0);
    x(0, 0) = true;
    x(1, 2) = true;

    ImArray<bool> x_out(3, 4);
    x_out.setConstant(0);

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

    EXPECT_TRUE((out == out_expected).all())
        << out;
}

TEST(SparseImage, correlate__upper_left_corner)
{
    ImArray<bool> x(3, 4);
    x.setConstant(0);
    x(0, 0) = 1;

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
        {5, 6, 0, 0},
        {8, 9, 0, 0},
        {0, 0, 0, 0}
    };

    EXPECT_TRUE((out == out_expected).all())
        << out;
}

TEST(SparseImage, CWiseBinaryOp)
{
    ImArray<bool> a
    {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0}

    };
    ImArray<bool> b
    {
        {1, 0, 1, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 0}
    };

    SparseImage<bool> sparse_image_a(a);

    auto operation = BinaryOp(
        sparse_image_a,
        b,
        ops::binary::boolean::And()
    );

    ImArray<bool> out = eval(operation);

    EXPECT_TRUE((out == (a && b)).all())
        << out;
}

TEST(SparseImage, CWiseOp)
{
    ImArray<bool> a
    {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0}

    };
    ImArray<bool> b
    {
        {1, 0, 1, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 0}

    };

    SparseImage<bool> sparse_image_a(a);
    SparseImage<bool> sparse_image_b(b);

    auto operation_both_sparse = BinaryOp(
        sparse_image_a,
        sparse_image_b,
        ops::binary::boolean::And());

    ImArray<bool> out_both_sparse(3, 4);
    out_both_sparse.setConstant(0);
    eval_to(out_both_sparse, operation_both_sparse);

    auto operation_lhs_sparse = BinaryOp(
        sparse_image_a,
        b,
        ops::binary::boolean::And());
    ImArray<bool> out_lhs_sparse(3, 4);
    out_lhs_sparse.setConstant(0);
    eval_to(out_lhs_sparse, operation_lhs_sparse);

    auto operation_rhs_sparse = BinaryOp(
        a,
        sparse_image_b,
        ops::binary::boolean::And());
    ImArray<bool> out_rhs_sparse(3, 4);
    out_rhs_sparse.setConstant(0);
    eval_to(out_rhs_sparse, operation_rhs_sparse);

    ImArray<bool> out_expected = a && b;

    EXPECT_TRUE((out_both_sparse == out_expected).all())
        << out_both_sparse;
    EXPECT_TRUE((out_lhs_sparse == out_expected).all())
        << out_lhs_sparse;
    EXPECT_TRUE((out_rhs_sparse == out_expected).all()) 
        << out_rhs_sparse; 
}
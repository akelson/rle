#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "imgproc/sparse_image.h"
#include <memory>

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using Eigen::Index;

template <typename PlainObjectType, typename Allocator = std::allocator<typename PlainObjectType::Scalar>>
struct MapStorage
{
    MapStorage(Eigen::Index rows, Eigen::Index cols, const Allocator &alloc) :
        buff(rows * cols, alloc),
        map(buff.data(), rows, cols) {};

    std::vector<typename PlainObjectType::Scalar, Allocator> buff;
    Eigen::Map<PlainObjectType> map;
};

TEST(MapStorage, ctor)
{
    MapStorage<Array<float, Dynamic, Dynamic>> storage(480, 640, std::allocator<int>());

    EXPECT_EQ(storage.map.cols(), 640);
    EXPECT_EQ(storage.map.rows(), 480);
    EXPECT_EQ(storage.buff.size(), 640 * 480);
}

TEST(BinaryOp, eval_scalar_operand)
{
    ScalarOperand<int> o(1);

    EXPECT_EQ(eval(o), 1);
}

TEST(BinaryOp, eval_scalar_addition)
{
    // 1 + 2 = 3
    ScalarOperand<int> lhs(1);
    ScalarOperand<int> rhs(2);

    BinaryOp<ScalarOperand<int>, ScalarOperand<int>, ops::binary::arithmetic::Addition> 
        operation{lhs, rhs, ops::binary::arithmetic::Addition()};

    EXPECT_EQ(eval(operation), 3);
}

TEST(BinaryOp, eval_scalar_subtraction)
{
    // 1 - 2 = -1
    auto lhs = ScalarOperand(1);
    auto rhs = ScalarOperand(2);

    auto operation = BinaryOp(lhs, rhs, ops::binary::arithmetic::Subtraction());

    EXPECT_EQ(eval(operation), -1);
}

TEST(BinaryOp, eval_nested_operation)
{
    // 1 + (2 * 3)
    auto operation = BinaryOp(
        ScalarOperand(1),
        BinaryOp(
            ScalarOperand(2),
            ScalarOperand(3),
            ops::binary::arithmetic::Multiplication()
        ),
        ops::binary::arithmetic::Addition()
    );
    
    EXPECT_EQ(eval(operation), 7);
}

TEST(IteratedBinaryOp, Summation)
{
    std::vector<int> values{1, 2, 3, 4, 5};

    ops::iterated_binary::Summation op;

    EXPECT_EQ(op(values), 15);
}

TEST(IteratedBinaryOp, Product)
{
    std::vector<int> values{1, 2, 3, 4, 5};

    ops::iterated_binary::Product op;

    EXPECT_EQ(op(values), 120);
}
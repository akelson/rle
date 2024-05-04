#pragma once

#include "imgproc/ops.h"
#include "imgproc/sparse_image.h"
#include <Eigen/Dense>

template <typename T_Op>
struct cwise_read_multiplier
{
    static const size_t value = 1;
};

template <typename T_Op>
struct cwise_dynamic_memory_required
{
    static const size_t value = 0;
};

template <typename T>
requires std::is_arithmetic_v<T>
struct ScalarOperand
{
    ScalarOperand(T value) : value(value) {}
    T value;
};

template <typename T>
struct RefOperand
{
    RefOperand(const T &value) : value(value) {}
    const T &value;
};

template <typename T_Lhs, typename T_Rhs, typename T_Op>
struct BinaryOp
{
    BinaryOp(T_Lhs lhs, T_Rhs rhs, T_Op op) : lhs(lhs), rhs(rhs), op(op) {}
    T_Lhs lhs;
    T_Rhs rhs;
    T_Op op;
};

template <typename T>
requires std::is_arithmetic_v<T>
T eval(ScalarOperand<T> operand)
{
    return operand.value;
}

template <typename T_Lhs, typename T_Rhs, typename T_Op>
auto eval(const BinaryOp<T_Lhs, T_Rhs, T_Op> &binary_op)
{
    return binary_op.op(eval(binary_op.lhs), eval(binary_op.rhs));
}

template <typename T_LhsPx, typename T_Rhs, typename T_Op>
auto eval(const BinaryOp<SparseImage<T_LhsPx>, T_Rhs, T_Op> &binary_op)
{
    const SparseImage<T_LhsPx> &lhs = binary_op.lhs;

    assert(lhs.height() == binary_op.rhs.rows());
    assert(lhs.width() == binary_op.rhs.cols());

    ImArray<T_LhsPx> out(lhs.height(), lhs.width());

    // A sparse image should contain mostly zeros.
    // Initialize the output to the result of the operation if the LHS was zero.
    out = binary_op.op(ImArray<T_LhsPx>::Zero(lhs.height(), lhs.width()), binary_op.rhs);

    visit_sparse_image(lhs, [&](Eigen::Index u, Eigen::Index v, bool lhs_val) {
        out(v, u) = binary_op.op(lhs_val, binary_op.rhs(v, u));
    });

    return out;
}

template <typename T, typename BinaryOpType>
void eval_to(ImArray<T> &out, const BinaryOpType &binary_op)
{
    using Eigen::Index;

    const auto &lhs = binary_op.lhs;
    const auto &rhs = binary_op.rhs;

    auto it_a = lhs.begin();
    auto it_b = rhs.begin();

    auto op = binary_op.op;

    // Iterate over the pixels in both a and b.
    // Only advance the iterator for the image with the smallest index.
    // After advancing an iterator, check if the indicies for both images are the same.
    // If they are the same set an output value to the result of the operation.
    // If they are not the same, set and output value to the result of the operation and a default value.
    while (it_a != lhs.end() && it_b != rhs.end())
    {
        if (it_a.index() < it_b.index())
        {
            const Index u = lhs.u(it_a);
            const Index v = lhs.v(it_a);
            out(v, u) = op(*it_a, T{});
            ++it_a;
        }
        else if (it_b.index() < it_a.index())
        {
            const Index u = rhs.u(it_a);
            const Index v = rhs.v(it_a);
            out(v, u) = op(T{}, *it_b);
            ++it_b;
        }
        else
        {
            const Index u = lhs.u(it_a);
            const Index v = lhs.v(it_a);
            out(v, u) = op(*it_a, *it_b);
            ++it_a;
            ++it_b;
        }
    }
}

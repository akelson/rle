#pragma once

#include "imgproc/ops.h"
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

template <typename T_BinaryOp>
auto eval(const T_BinaryOp &binary_op)
{
    return binary_op.op(eval(binary_op.lhs), eval(binary_op.rhs));
}

template <typename Op, typename LhsType, typename RhsType>
struct CWiseOp
{
    using OpType = Op;

    const LhsType lhs_;
    const RhsType rhs_;
};

template <typename T, typename CWiseOpType>
void eval_to(ImArray<T> &out, const CWiseOpType &op)
{
    using Eigen::Index;

    auto it_a = op.lhs_.begin();
    auto it_b = op.rhs_.begin();

    using Op = typename CWiseOpType::OpType;

    // Iterate over the pixels in both a and b.
    // Only advance the iterator for the image with the smallest index.
    // After advancing an iterator, check if the indicies for both images are the same.
    // If they are the same set an output value to the result of the operation.
    // If they are not the same, set and output value to the result of the operation and a default value.
    while (it_a != op.lhs_.end() && it_b != op.rhs_.end())
    {
        if (it_a.index() < it_b.index())
        {
            const Index u = op.lhs_.u(it_a);
            const Index v = op.lhs_.v(it_a);
            out(v, u) = Op()(*it_a, T{});
            ++it_a;
        }
        else if (it_b.index() < it_a.index())
        {
            const Index u = op.rhs_.u(it_a);
            const Index v = op.rhs_.v(it_a);
            out(v, u) = Op()(T{}, *it_b);
            ++it_b;
        }
        else
        {
            const Index u = op.lhs_.u(it_a);
            const Index v = op.lhs_.v(it_a);
            out(v, u) = Op()(*it_a, *it_b);
            ++it_a;
            ++it_b;
        }
    }
}

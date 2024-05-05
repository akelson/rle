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

template <typename DerivedOut, typename T_LhsPx, typename T_Rhs, typename T_Op>
auto eval_to(Eigen::ArrayBase<DerivedOut> &out, const BinaryOp<SparseImage<T_LhsPx>, T_Rhs, T_Op> &binary_op)
{
    const SparseImage<T_LhsPx> &lhs = binary_op.lhs;
    const auto &rhs = binary_op.rhs;

    assert(lhs.height() == rhs.rows());
    assert(lhs.width() == rhs.cols());
    assert(out.rows() == rhs.rows());
    assert(out.cols() == rhs.cols());

    // A sparse image should contain mostly zeros.
    // Initialize the output to the result of the operation if the LHS was zero.
    out = binary_op.op(ImArray<T_LhsPx>::Zero(lhs.height(), lhs.width()), rhs);

    visit_sparse_image(lhs, [&](Eigen::Index u, Eigen::Index v, bool lhs_val) {
        out(v, u) = binary_op.op(lhs_val, rhs(v, u));
    });
}

template <typename DerivedOut, typename T_Lhs, typename T_RhsPx, typename T_Op>
auto eval_to(Eigen::ArrayBase<DerivedOut> &out, const BinaryOp<T_Lhs, SparseImage<T_RhsPx>, T_Op> &binary_op)
{
    const auto &lhs = binary_op.lhs;
    const SparseImage<T_RhsPx> &rhs = binary_op.rhs;

    assert(lhs.rows() == rhs.height());
    assert(lhs.cols() == rhs.width());
    assert(out.rows() == rhs.height());
    assert(out.cols() == rhs.width());

    // A sparse image should contain mostly zeros.
    // Initialize the output to the result of the operation if the RHS was zero.
    out = binary_op.op(lhs, ImArray<T_RhsPx>::Zero(rhs.height(), rhs.width()));

    visit_sparse_image(rhs, [&](Eigen::Index u, Eigen::Index v, bool rhs_val) {
        out(v, u) = binary_op.op(lhs(v, u), rhs_val);
    });
}

template <typename T_LhsPx, typename T_Rhs, typename T_Op>
auto eval(const BinaryOp<SparseImage<T_LhsPx>, T_Rhs, T_Op> &binary_op)
{
    ImArray<T_LhsPx> out(binary_op.lhs.rows(), binary_op.lhs.cols());
    eval_to(out, binary_op);
    return out;
}

template <typename T_LhsPx, typename T_RhsPx, typename T_Op>
auto eval_to(ImArray<T_LhsPx> &out, const BinaryOp<SparseImage<T_LhsPx>, SparseImage<T_RhsPx>, T_Op> &binary_op)
{
    using Eigen::Index;

    const auto &lhs = binary_op.lhs;
    const auto &rhs = binary_op.rhs;
    auto op = binary_op.op;
    const auto default_value = T_LhsPx{};

    auto it_lhs = lhs.begin();
    auto it_rhs = rhs.begin();

    while (it_lhs != lhs.end() && it_rhs != rhs.end())
    {
        if (it_lhs.index() < it_rhs.index())
        {
            const Index u = lhs.u(it_lhs);
            const Index v = lhs.v(it_lhs);
            out(v, u) = op(*it_lhs, default_value);
            ++it_lhs;
        }
        else if (it_rhs.index() < it_lhs.index())
        {
            const Index u = rhs.u(it_rhs);
            const Index v = rhs.v(it_rhs);
            out(v, u) = op(default_value, *it_rhs);
            ++it_rhs;
        }
        else
        {
            const Index u = lhs.u(it_lhs);
            const Index v = lhs.v(it_lhs);
            out(v, u) = op(*it_lhs, *it_rhs);
            ++it_lhs;
            ++it_rhs;
        }
    }
}

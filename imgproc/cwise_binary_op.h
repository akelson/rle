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
    using value_type = T;
    ScalarOperand(T value) : value(value) {}
    T eval() const { return value; }
    template <typename Allocator>
    T eval(Allocator &) const { return value; }
    T value;
};

template <typename T>
struct RefOperand
{
    using value_type = T;
    RefOperand(const T &value) : value(value) {}
    const T& eval() const { return value; }
    template <typename Allocator>
    const T& eval(Allocator &) const { return value; }
    const T &value;
};

template <typename T_Lhs, typename T_Rhs, typename T_Op>
struct BinaryOpBase
{
    using value_type = typename std::remove_reference_t<T_Lhs>::value_type;
    T_Lhs lhs;
    T_Rhs rhs;
    T_Op op;
};

template <typename T_Lhs, typename T_Rhs, typename T_Op>
struct BinaryOp : public BinaryOpBase<T_Lhs, T_Rhs, T_Op>
{
    using Base = BinaryOpBase<T_Lhs, T_Rhs, T_Op>;
    BinaryOp(T_Lhs lhs, T_Rhs rhs, T_Op op) : Base{lhs, rhs, op} {}

    template <typename Derived>
    auto eval_to(Eigen::ArrayBase<Derived> &out) const
    {
        out = eval();
    }

    template <typename Derived, typename Allocator>
    auto eval_to(Eigen::ArrayBase<Derived> &out, Allocator alloc) const
    {
        out = eval(alloc);
    }

    auto eval() const 
    {
        return Base::op(Base::lhs.eval(), Base::rhs.eval());
    }

    template <typename Allocator>
    auto eval(Allocator &alloc) const
    { 
        return Base::op(Base::lhs.eval(alloc), Base::rhs.eval(alloc));
    }
};

template <typename T_Op, typename Allocator>
auto eval_to_imarray(const T_Op &op, Allocator &alloc)
{
    // TODO: Use custom allocator
    ImArray<typename T_Op::value_type> out(op.lhs.rows(), op.rhs.cols());
    op.eval_to(out, alloc);
    return out;
}

template <typename T_LhsPx, typename T_RhsPx, typename T_Op>
struct BinaryOp<SparseImage<T_LhsPx>, SparseImage<T_RhsPx>, T_Op> : 
    public BinaryOpBase<const SparseImage<T_LhsPx>&, const SparseImage<T_RhsPx>&, T_Op>
{
    using value_type = T_LhsPx;
    using T_Lhs = const SparseImage<T_LhsPx>&;
    using T_Rhs = const SparseImage<T_RhsPx>&;
    using Base = BinaryOpBase<T_Lhs, T_Rhs, T_Op>;

    BinaryOp(const SparseImage<T_LhsPx> &lhs, const SparseImage<T_RhsPx> &rhs, T_Op op) : Base{lhs, rhs, op} {}

    template <typename Derived, typename Allocator>
    auto eval_to(Eigen::ArrayBase<Derived> &out, Allocator alloc) const
    {
        using Eigen::Index;

        const auto &lhs = Base::lhs;
        const auto &rhs = Base::rhs;
        auto op = Base::op;
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

    template <typename Allocator>
    auto eval(Allocator &alloc) const
    { 
        return eval_to_imarray(*this, alloc);
    }
};

template <typename T_LhsPx, typename T_Rhs, typename T_Op>
struct BinaryOp<SparseImage<T_LhsPx>, T_Rhs, T_Op> : 
    public BinaryOpBase<const SparseImage<T_LhsPx>&, T_Rhs, T_Op>
{
    using value_type = T_LhsPx;
    using T_Lhs = const SparseImage<T_LhsPx>&;
    using Base = BinaryOpBase<T_Lhs, T_Rhs, T_Op>;

    BinaryOp(const SparseImage<T_LhsPx> &lhs, T_Rhs rhs, T_Op op) : Base{lhs, rhs, op} {}

    template <typename Derived, typename Allocator>
    auto eval_to(Eigen::ArrayBase<Derived> &out, Allocator alloc) const
    {
        const SparseImage<T_LhsPx> &lhs = Base::lhs;
        const auto &rhs = Base::rhs;
        auto op = Base::op;

        assert(lhs.height() == rhs.rows());
        assert(lhs.width() == rhs.cols());
        assert(out.rows() == rhs.rows());
        assert(out.cols() == rhs.cols());

        // A sparse image should contain mostly zeros.
        // Initialize the output to the result of the operation if the LHS was zero.
        out = op(ImArray<T_LhsPx>::Zero(lhs.height(), lhs.width()), rhs);

        visit_sparse_image(lhs, [&](Eigen::Index u, Eigen::Index v, bool lhs_val) {
            out(v, u) = op(lhs_val, rhs(v, u));
        });
    }

    template <typename Allocator>
    auto eval(Allocator &alloc) const
    { 
        return eval_to_imarray(*this, alloc);
    }
};

template <typename T_Lhs, typename T_RhsPx, typename T_Op>
struct BinaryOp<T_Lhs, SparseImage<T_RhsPx>, T_Op> : 
    public BinaryOpBase<T_Lhs, const SparseImage<T_RhsPx>&, T_Op>
{
    using value_type = typename T_Lhs::value_type;
    using T_Rhs = const SparseImage<T_RhsPx>&;
    using Base = BinaryOpBase<T_Lhs, T_Rhs, T_Op>;

    BinaryOp(T_Lhs lhs, const SparseImage<T_RhsPx> &rhs, T_Op op) : Base{lhs, rhs, op} {}

    template <typename Derived, typename Allocator>
    auto eval_to(Eigen::ArrayBase<Derived> &out, Allocator alloc) const
    {
        const auto &lhs = Base::lhs;
        const SparseImage<T_RhsPx> &rhs = Base::rhs;
        auto op = Base::op;

        assert(lhs.rows() == rhs.height());
        assert(lhs.cols() == rhs.width());
        assert(out.rows() == rhs.height());
        assert(out.cols() == rhs.width());

        // A sparse image should contain mostly zeros.
        // Initialize the output to the result of the operation if the RHS was zero.
        out = op(lhs, ImArray<T_RhsPx>::Zero(rhs.height(), rhs.width()));

        visit_sparse_image(rhs, [&](Eigen::Index u, Eigen::Index v, bool rhs_val) {
            out(v, u) = op(lhs(v, u), rhs_val);
        });
    }

    template <typename Allocator>
    auto eval(Allocator &alloc) const
    { 
        return eval_to_imarray(*this, alloc);
    }
};
    
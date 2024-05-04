#pragma once

namespace ops
{

namespace binary::boolean
{
    struct And
    {
        template <typename T>
        T operator()(T a, T b) const { return a && b; }
    };

    struct Or
    {
        template <typename T>
        T operator()(T a, T b) const { return a || b; }
    };
} // binary::boolean

namespace binary::arithmetic
{
    struct Addition 
    {
        template <typename T>
        T operator()(T a, T b) const { return a + b; }
    };

    struct Subtraction
    {
        template <typename T>
        T operator()(T a, T b) const { return a - b; }
    };

    struct Multiplication
    {
        template <typename T>
        T operator()(T a, T b) const { return a * b; }
    };

    struct Division
    {
        template <typename T>
        T operator()(T a, T b) const { return a / b; }
    };
} // binary::arithmetic

namespace unary::arithmetic
{
    struct Negation
    {
        template <typename T>
        T operator()(T a) const { return -a; }
    };

    struct Increment
    {
        template <typename T>
        T operator()(T a) const { return a + 1; }
    };

    struct Decrement
    {
        template <typename T>
        T operator()(T a) const { return a - 1; }
    };
} // unary::arithmetic

namespace iterated_binary
{
    template <typename Op>
    struct generic_iterated_binary_op
    {
        Op op;

        template <typename T>
        auto operator()(const T& values) const
        {
            auto it = values.begin();

            using result_type = typename std::iterator_traits<decltype(it)>::value_type;

            if (it == values.end()) return result_type{};

            result_type result = *it;

            for (++it; it != values.end(); ++it)
            {
                result = op(result, *it);
            }

            return result;
        }
    };
    
    using Summation = generic_iterated_binary_op<binary::arithmetic::Addition>;
    using Product = generic_iterated_binary_op<binary::arithmetic::Multiplication>;
}

} // ops
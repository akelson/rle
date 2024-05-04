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

} // ops
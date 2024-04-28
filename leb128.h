#pragma once

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <span>

namespace codec::leb128
{

template <typename T>
requires std::integral<T>
std::span<uint8_t> encode(T val, std::span<uint8_t> buff)
{
    auto it = buff.begin();

    do
    {
        if (buff.end() == it)
        {
            throw std::runtime_error("buffer too small");
        }
        *(it++) = val & 0x7f;
        val >>= 7;
    } while (val);

    return std::span(buff.begin(), it);
}

template <typename T>
requires std::integral<T>
std::tuple<T, std::span<uint8_t>> decode_one(std::span<uint8_t> buff)
{
    T val{};
    auto it = buff.begin();

    do {
        const uint8_t code = *it++;
        const bool more_bytes = 0x80 & code;
        val += 0x7f & code;
        if (more_bytes) 
        {
            val <<= 7;
        }
    while (more_bytes)

    return {val, std::span<uint8_t>(buff.begin(), it)};
}

}
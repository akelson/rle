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
std::tuple<T, std::span<const uint8_t>> decode_one(std::span<const uint8_t> buff)
{
    T val{};
    auto it = buff.begin();

    bool more_bytes{};
    uint8_t val_shift = 0;
    do {
        if (it >= buff.end())
        {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        const uint8_t code = *it++;
        more_bytes = 0x80 & code;
        val += (0x7f & code) << val_shift;
        val_shift += 7;
    }
    while (more_bytes);

    return {val, std::span<const uint8_t>(buff.begin(), it)};
}

}
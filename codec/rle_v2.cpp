#include "codec/rle_v2.h"
#include "codec/leb128.h"
#include <stdexcept>
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>

namespace rle::v2 {

std::span<uint8_t> encode(std::span<const uint8_t> data, std::span<uint8_t> rle_buff)
{
    bool prev_val = false;
    uint32_t run_length = 0;
    auto rle_it = rle_buff.begin();
    for (const uint8_t val : data)
    {
        if (val != prev_val)
        {
            if (rle_buff.end() == rle_it)
            {
                throw std::runtime_error("rle_buff buffer too small");
            }

            std::span<uint8_t> encoded = codec::leb128::encode(run_length, std::span(rle_it, rle_buff.end()));
            rle_it = encoded.end();
            run_length = 1;
        } // end if
        else
        {
            ++run_length;
        }
        prev_val = val;
    } // end for val

    if (rle_buff.end() == rle_it)
    {
        throw std::runtime_error("rle_buff buffer too small");
    }

    std::span<uint8_t> encoded = codec::leb128::encode(run_length, std::span(rle_it, rle_buff.end()));
    rle_it = encoded.end();

    return std::span(rle_buff.begin(), rle_it);
}

std::span<uint8_t> decode(std::span<const uint8_t> rle_buff, std::span<uint8_t> data_buff)
{
    bool cur_val = 0;
    auto data_it = data_buff.begin();
    decltype(rle_buff)::iterator rle_it = rle_buff.begin();

    while (rle_buff.end() > rle_it)
    {
        const auto [run_length, decoded_buff] = codec::leb128::decode_one<int>(std::span(rle_it, rle_buff.end()));
        rle_it = decoded_buff.end();
        for (size_t i = 0; i < run_length; i++)
        {
            if (data_buff.end() == data_it)
            {
                throw std::runtime_error("data buffer too small");
            }
            *data_it++ = cur_val;
        }
        cur_val = !cur_val;
    } // end while

    while (data_buff.end() != data_it)
    {
        *data_it++ = cur_val;
    }

    return std::span(data_buff.begin(), data_it);
}

}
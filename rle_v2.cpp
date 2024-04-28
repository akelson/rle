#include "rle_v2.h"
#include <stdexcept>
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>

namespace rle::v1 {

std::span<uint8_t> encode(std::span<const uint8_t> data, std::span<uint8_t> rle_buff)
{
    bool prev_val = 0;
    int run_length = 0;
    auto rle_it = rle_buff.begin();
    for (const uint8_t val : data)
    {
        if (val != prev_val)
        {
            if (rle_buff.end() == rle_it)
            {
                throw std::runtime_error("rle_buff buffer too small");
            }

            *rle_it++ = static_cast<uint8_t>(run_length);
            run_length = 0;
        } // end if
        else
        {
            ++run_length;
        }
        prev_val = val;
    } // end for val
    return std::span(rle_buff.begin(), rle_it);
}

std::span<uint8_t> decode(std::span<const uint8_t> rle, std::span<uint8_t> data_buff)
{
    return {};
}

}
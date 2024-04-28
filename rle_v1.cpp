#include "rle_v1.h"
#include <stdexcept>
#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

namespace rle::v1 {

static const std::vector<int> make_skip_table()
{
    const size_t N = 16;
    const int max = 10e6;
    const int min = 0xff - N;
    std::vector<int> skip_table;
    for (int i = 0; i < N; i++)
    {
        int ii = N - i - 1;
        float base = std::exp(std::log(max/min)/N);
        //int val = ii * float(max - min) / N + min;
        long int val = std::pow(base, ii) * min;
        //assert(val < std::numeric_limits::<int>::max);
        skip_table.push_back(val);
    }
    std::cout << std::endl;
    return skip_table;
}

static const std::vector<int> skip_table = make_skip_table();

std::span<uint8_t> encode(std::span<const uint8_t> data, std::span<uint8_t> rle_buff)
{
    bool prev_val = 0;
    int run_length = 0;
    auto rle_it = rle_buff.begin();
    for (const uint8_t val : data)
    {
        if (val != prev_val)
        {
            for (size_t skip_table_i = 0; skip_table_i < skip_table.size(); skip_table_i++)
            {
                int skip_val = skip_table[skip_table_i];
                while (run_length > skip_val)
                {
                    if (rle_buff.end() == rle_it)
                    {
                        throw std::runtime_error("rle_buff buffer too small");
                    }

                    *rle_it++ = 0xff - skip_table_i;
                    run_length -= skip_val;
                }
            } // end for skip_table_i

            if (rle_buff.end() == rle_it)
            {
                throw std::runtime_error("rle_buff buffer too small");
            }

            assert(run_length < 0xff - (skip_table.size() - 1));

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
    bool cur_val = 0;
    auto data_it = data_buff.begin();
    for (const uint8_t rle_val : rle)
    {
        const size_t skip_table_i = 0xff - rle_val;
        if (skip_table_i < skip_table.size())
        {
            const int skip_val = skip_table[skip_table_i];
            data_it += skip_val;
        }
        else
        {
            for (size_t i = 0; i < rle_val; i++)
            {
                if (data_buff.end() == data_it)
                {
                    throw std::runtime_error("data buffer too small");
                }
                *data_it++ = cur_val;
            }
            cur_val = !cur_val;
            if (data_buff.end() == data_it)
            {
                throw std::runtime_error("data buffer too small");
            }
            *data_it++ = cur_val;
        }
    } // end for

    while (data_buff.end() != data_it)
    {
        *data_it++ = cur_val;
    }

    return std::span(data_buff.begin(), data_it);
}

}
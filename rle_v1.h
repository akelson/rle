#pragma once

#include <span>
#include <cstdint>

namespace rle::v1 {

std::span<uint8_t> encode(std::span<const uint8_t> data, std::span<uint8_t> rle_buff);
std::span<uint8_t> decode(std::span<const uint8_t> rle, std::span<uint8_t> data_buff);

}
#include "leb128.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>

using ::testing::ElementsAre;
using namespace codec;

TEST(LEB128, encode)
{
    std::vector<uint8_t> buff(3);
    std::span<uint8_t> encoded;

    encoded = leb128::encode(uint32_t(0), buff);
    EXPECT_THAT(encoded, ElementsAre(0));
    EXPECT_THAT(buff, ElementsAre(0, 0, 0));

    encoded = leb128::encode(uint32_t(1), buff);
    EXPECT_THAT(encoded, ElementsAre(1));
    EXPECT_THAT(buff, ElementsAre(1, 0, 0));

    encoded = leb128::encode(uint32_t(127), buff);
    EXPECT_THAT(encoded, ElementsAre(127));
    EXPECT_THAT(buff, ElementsAre(127, 0, 0));

    encoded = leb128::encode(uint32_t(128), buff);
    EXPECT_THAT(encoded, ElementsAre(0x80, 0x01));
    EXPECT_THAT(buff, ElementsAre(0x80, 0x01, 0x00));

    encoded = leb128::encode(uint32_t(129), buff);
    EXPECT_THAT(encoded, ElementsAre(0x81, 0x01));
    EXPECT_THAT(buff, ElementsAre(0x81, 0x01, 0x00));

    encoded = leb128::encode(uint8_t(130), buff);
    EXPECT_THAT(encoded, ElementsAre(0x82, 0x01));
    EXPECT_THAT(buff, ElementsAre(0x82, 0x01, 0x00));

    buff.resize(5, 0);
    encoded = leb128::encode(uint64_t(0x100000000), buff);
    EXPECT_THAT(encoded, ElementsAre(0x80, 0x80, 0x80, 0x80, 0x10));
    EXPECT_THAT(buff, ElementsAre(0x80, 0x80, 0x80, 0x80, 0x10));

    encoded = leb128::encode(uint32_t(0xb108c3f1), buff);
    EXPECT_THAT(encoded, ElementsAre(0xf1, 0x87, 0xa3, 0x88, 0x0b));
}

TEST(LEB128, decode_one)
{
    std::vector<uint8_t> encoded;
    uint32_t val{};
    std::span<const uint8_t> encoded_val_buff;

    encoded = {1};
    std::tie(val, encoded_val_buff) = leb128::decode_one<uint32_t>(encoded);
    EXPECT_EQ(val, 1);
    EXPECT_THAT(encoded_val_buff, ElementsAre(1));

    encoded = {127};
    std::tie(val, encoded_val_buff) = leb128::decode_one<uint32_t>(encoded);
    EXPECT_EQ(val, 127);
    EXPECT_THAT(encoded_val_buff, ElementsAre(127));

    encoded = {0xf1, 0x87, 0xa3, 0x88, 0x0b};
    std::tie(val, encoded_val_buff) = leb128::decode_one<uint32_t>(encoded);
    EXPECT_EQ(val, 2970141681);
}

TEST(LEB128, encode_decode)
{
    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<uint32_t> uniform_dist;

    std::vector<uint8_t> buff(5);

    for (int i = 0; i < 1000; i++)
    {
        const uint32_t val = uniform_dist(e1);
        auto encoded = leb128::encode(val, buff);
        auto [decoded_val, encoded_val_buff] = leb128::decode_one<int>(encoded);
        EXPECT_EQ(decoded_val, val);
    }
}
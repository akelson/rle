#include "leb128.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

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
    EXPECT_THAT(encoded, ElementsAre(0, 1));
    EXPECT_THAT(buff, ElementsAre(0, 1, 0));

    encoded = leb128::encode(uint32_t(129), buff);
    EXPECT_THAT(encoded, ElementsAre(1, 1));
    EXPECT_THAT(buff, ElementsAre(1, 1, 0));

    encoded = leb128::encode(uint8_t(130), buff);
    EXPECT_THAT(encoded, ElementsAre(2, 1));
    EXPECT_THAT(buff, ElementsAre(2, 1, 0));

    encoded = leb128::encode(uint8_t(130), buff);
    EXPECT_THAT(encoded, ElementsAre(2, 1));
    EXPECT_THAT(buff, ElementsAre(2, 1, 0));

    buff.resize(5, 0);
    encoded = leb128::encode(uint64_t(0x100000000), buff);
    EXPECT_THAT(encoded, ElementsAre(0, 0, 0, 0, 0x10));
    EXPECT_THAT(buff, ElementsAre(0, 0, 0, 0, 0x10));
}
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "rle.h"
#include <span>
#include <Eigen/Dense>

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using ::testing::ElementsAre;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

TEST(rle, encode)
{
    Vector<uint8_t, Dynamic> x(10);
    x.setConstant(0);

    x(0) = 1;
    x(3) = 1;

    std::vector<uint8_t> buff(1024);
    std::span<uint8_t> rle = encode(std::span(x.data(), x.size()), buff);

    ASSERT_THAT(rle, ElementsAre(0, 0, 1, 0));
}

TEST(rle, encode_decode)
{
    Array<float, Dynamic, Dynamic> rand_x = Matrix<float, Dynamic, Dynamic>::Random(1024, 768);
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x = (rand_x > 0.9).cast<uint8_t>();

    x.block(500, 500, 200, 100) = 1;
    x.block(5, 5, 2, 2) = 1;

    std::vector<uint8_t> buff(10e6);
    std::span<uint8_t> rle = encode(std::span(x.data(), x.size()), buff);

    Map<Vector<uint8_t, Dynamic>> rle_map(rle.data(), rle.size());

    Array<uint8_t, Dynamic, Dynamic, RowMajor> x_decoded;
    x_decoded.resizeLike(x);

    decode(rle, std::span(x_decoded.data(), x_decoded.size()));

    EXPECT_TRUE((x == x_decoded).all());
}
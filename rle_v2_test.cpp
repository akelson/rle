#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "rle_v2.h"
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

using namespace rle::v2;

TEST(rle_v2, encode)
{
    Vector<uint8_t, Dynamic> x(10);
    x.setConstant(0);

    x(0) = 1;
    x(3) = 1;

    std::vector<uint8_t> buff(1024);
    std::span<uint8_t> encoded = encode(std::span(x.data(), x.size()), buff);

    ASSERT_THAT(encoded, ElementsAre(0, 0, 1, 0));
}

TEST(rle_v2, encode_decode)
{
    Array<float, Dynamic, Dynamic> rand_x = Matrix<float, Dynamic, Dynamic>::Random(2048, 2000);
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x = (rand_x > 0.95).cast<uint8_t>();

    std::vector<uint8_t> buff(10e6);
    std::span<uint8_t> encoded = encode(std::span(x.data(), x.size()), buff);
    DISP(encoded.size());
    DISP(float(x.size()/8)/encoded.size());

    Map<Vector<uint8_t, Dynamic>> rle_map(encoded.data(), encoded.size());

    Array<uint8_t, Dynamic, Dynamic, RowMajor> x_decoded;
    x_decoded.resizeLike(x);

    decode(encoded, std::span(x_decoded.data(), x_decoded.size()));

    EXPECT_TRUE((x == x_decoded).all());
}

TEST(rle_v2, encode_decode_sparse)
{
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x(2048, 2000);
    x.setConstant(0);

    x.block(500, 500, 200, 100) = 1;
    x.block(5, 5, 2, 2) = 1;

    std::vector<uint8_t> buff(10e6);
    std::span<uint8_t> encoded = encode(std::span(x.data(), x.size()), buff);
    DISP(encoded.size());
    DISP(float(x.size()/8)/encoded.size());

    Map<Vector<uint8_t, Dynamic>> rle_map(encoded.data(), encoded.size());

    Array<uint8_t, Dynamic, Dynamic, RowMajor> x_decoded;
    x_decoded.resizeLike(x);

    decode(encoded, std::span(x_decoded.data(), x_decoded.size()));

    EXPECT_TRUE((x == x_decoded).all());
}

TEST(rle_v2, encode_decode_sparse_small)
{
    Array<uint8_t, Dynamic, Dynamic, RowMajor> x(640, 480);
    x.setConstant(0);
    x.block(100, 200, 50, 40) = 1;
    x.block(5, 5, 2, 2) = 1;

    std::vector<uint8_t> buff(10e6);
    std::span<uint8_t> encoded = encode(std::span(x.data(), x.size()), buff);
    DISP(encoded.size());
    DISP(float(x.size()/8)/encoded.size());

    Map<Vector<uint8_t, Dynamic>> rle_map(encoded.data(), encoded.size());

    Array<uint8_t, Dynamic, Dynamic, RowMajor> x_decoded;
    x_decoded.resizeLike(x);

    decode(encoded, std::span(x_decoded.data(), x_decoded.size()));

    EXPECT_TRUE((x == x_decoded).all());
}
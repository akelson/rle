#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "imgproc/sparse_image.h"
#include <memory>

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using Eigen::Index;

template <typename PlainObjectType, typename Allocator = std::allocator<typename PlainObjectType::Scalar>>
struct MapStorage
{
    MapStorage(Eigen::Index rows, Eigen::Index cols, const Allocator &alloc) :
        buff(rows * cols, alloc),
        map(buff.data(), rows, cols) {};

    std::vector<typename PlainObjectType::Scalar, Allocator> buff;
    Eigen::Map<PlainObjectType> map;
};

TEST(MapStorage, ctor)
{
    MapStorage<Array<float, Dynamic, Dynamic>> storage(480, 640, std::allocator<int>());

    EXPECT_EQ(storage.map.cols(), 640);
    EXPECT_EQ(storage.map.rows(), 480);
    EXPECT_EQ(storage.buff.size(), 640 * 480);
}
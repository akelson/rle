#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "sparse_image.h"

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Map;
using Eigen::Vector;
using Eigen::RowMajor;
using ::testing::ElementsAre;

#define DISP(x) std::cout << #x << ":\n" << x << std::endl;

TEST(SparseImage, test_1)
{
    Array<bool, Dynamic, Dynamic, RowMajor> x(2, 3);
    x.setConstant(0);
    x(0, 0) = true;
    x(1, 1) = true;

    SparseImage<bool> sparse_image(x);

    EXPECT_EQ(sparse_image.width(), x.cols());
    EXPECT_EQ(sparse_image.height(), x.rows());
    EXPECT_EQ(sparse_image.size(), x.rows() * x.cols());

    int i = 0;
    for(auto it = sparse_image.begin(); it != sparse_image.end(); ++it)
    {
        DISP(it.index());
    }
}
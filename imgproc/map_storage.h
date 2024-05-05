#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

template <typename PlainObjectType, typename Allocator = std::allocator<typename PlainObjectType::Scalar>>
struct MapStorage
{

    MapStorage(Eigen::Index rows, Eigen::Index cols, Allocator &alloc) :
        buff(
            alloc.allocate(rows * cols), 
            [&alloc, rows, cols](auto p){ alloc.deallocate(p, rows * cols); }),
        map(buff.get(), rows, cols) {};

    std::unique_ptr<
        typename PlainObjectType::Scalar[],
        std::function<void(typename PlainObjectType::Scalar[])>> buff;
    Eigen::Map<PlainObjectType> map;
};

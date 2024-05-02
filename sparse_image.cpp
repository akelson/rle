#include "sparse_image.h"
#include <span>

template<>
SparseImage<bool>::SparseImage(const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &array) :
    width_(array.cols()),
    height_(array.rows()),
    encoded_runs_()
{
    // TODO: Count the runs and allocate the right amount of memory.
    owned_buff_.resize(10e6);

    encoded_runs_ = rle::v2::encode(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(array.data()), array.size()), owned_buff_);
}
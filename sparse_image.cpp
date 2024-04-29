#include "sparse_image.h"
#include <span>

template<>
SparseImage<bool>::SparseImage(Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> array) :
    width_(array.cols()),
    height_(array.rows()),
    encoded_runs_()
{
    // TODO: Count the runs and allocate the right amount of memory.
    owned_buff_.resize(10e6);

    rle::v2::encode(std::span<uint8_t>(reinterpret_cast<uint8_t*>(array.data()), array.size()), owned_buff_);

    encoded_runs_ = owned_buff_;
}
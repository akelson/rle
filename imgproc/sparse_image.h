#pragma once

#include "imgproc/common.h"
#include "codec/leb128.h"
#include "codec/rle_v2.h"
#include <Eigen/Dense>
#include <concepts>
#include <iterator>

inline std::tuple<bool, uint32_t, std::span<const uint8_t>> decode_run(std::span<const uint8_t> buff, bool prev_value)
{
    auto [run_length, encoded_run_length] = codec::leb128::decode_one<uint32_t>(buff);
    return {!prev_value, run_length, encoded_run_length};
}

template <typename T>
requires std::is_arithmetic_v<T>
class PixelIterator
{
  public:

    PixelIterator(T value, size_t index, size_t num_px, std::span<const uint8_t> encoded_runs) : 
        value_(value),
        index_(index),
        num_px_(num_px),
        run_length_(),
        remaining_encoded_runs_(encoded_runs)
    {
        while (0 == run_length_ && !remaining_encoded_runs_.empty())
        {
            auto [new_value, decoded_run_length, encoded_run_length] = decode_run(remaining_encoded_runs_, value_);
            value_ = new_value;
            run_length_ = decoded_run_length;
            remaining_encoded_runs_ = std::span<const uint8_t>(encoded_run_length.end(), encoded_runs.end());
        }

        // Skip runs of zero
        if (0 == value_ && 0 != run_length_)
        {
            index_ += run_length_ - 1;
            run_length_ = 1;
            ++(*this);
        }
    }

    // End iterator
    PixelIterator(size_t num_px) :
        value_(),
        index_(num_px),
        num_px_(num_px),
        run_length_(),
        remaining_encoded_runs_() {}

    PixelIterator& operator++()
    {
        ++index_;
        --run_length_;

        if (0 == run_length_)
        {
            *this = PixelIterator(value_, index_, num_px_, remaining_encoded_runs_);
        }

        return *this;
    }

    bool operator==(const PixelIterator& rhs) const { return index_ == rhs.index_; }
    bool operator!=(const PixelIterator& rhs) const { return !(*this == rhs); }
    const T& operator*() const { return value_; }

    size_t index() const { return index_; }

  private:
    T value_;
    size_t index_;
    size_t num_px_;
    size_t run_length_;
    size_t stride_;
    std::span<const uint8_t> remaining_encoded_runs_;
};


template <typename T>
requires std::is_arithmetic_v<T>
bool init_prev_value(); 

template <>
inline bool init_prev_value<bool>() { return true; }

template <typename T>
requires std::is_arithmetic_v<T>
class SparseImage
{
  public:
    SparseImage(Eigen::Index width, Eigen::Index height, std::span<uint8_t> encoded_runs) :
        width_(width),
        height_(height),
        encoded_runs_(encoded_runs) 
    {
    }

    SparseImage(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &array);

    PixelIterator<T> begin() const
    {
        return PixelIterator(init_prev_value<bool>(), 0, size(), encoded_runs_);
    }

    PixelIterator<T> end() const
    {
        return PixelIterator<T>(size());
    }

    Eigen::Index width() const { return width_; }
    Eigen::Index height() const { return height_; }
    Eigen::Index size() const { return width_ * height_; }

    Eigen::Index rows() const { return height_; }
    Eigen::Index cols() const { return width_; }

    Eigen::Index u(const PixelIterator<T> &it) const { return it.index() % width_; }
    Eigen::Index v(const PixelIterator<T> &it) const { return it.index() / width_; }

  private:
    std::span<uint8_t> encoded_runs_;
    Eigen::Index width_;
    Eigen::Index height_;
    std::vector<uint8_t> owned_buff_;
};

inline void visit_sparse_image(
    const SparseImage<bool>& image,
    std::function<void(Eigen::Index, Eigen::Index, bool)> visitor)
{
    for (auto it = image.begin(); it != image.end(); ++it)
    {
        const auto u = image.u(it);
        const auto v = image.v(it);
        const auto val = *it;
        visitor(u, v, val);
    }
}

template <typename T=float>
requires std::is_arithmetic_v<T>
void from_sparse_image(const SparseImage<T>& image, ImArrayRef<T> out)
{
    visit_sparse_image(image, [&out](Eigen::Index u, Eigen::Index v, T val) {
        out(v, u) = val;
    });
}

template <typename T=float>
requires std::is_arithmetic_v<T>
void correlate(const SparseImage<bool>& image, const ImArrayRef<T> &kernel, ImArrayRef<T> out)
{
    using Eigen::Index;
    using Eigen::Vector;

    assert(kernel.cols() % 2 == 1);
    assert(kernel.rows() % 2 == 1);

    const Vector<Index, 2> img_size(out.cols(), out.rows());
    const Vector<Index, 2> kernel_size(kernel.cols(), kernel.rows());
    const Vector<Index, 2> half_kernel_size = kernel_size / 2;

    visit_sparse_image(image, [&](Eigen::Index u, Eigen::Index v, bool val) {
        // UV coordinates in the input image
        const Vector<Index, 2> uv_img(u, v);

        // Min and max UV coordinates in the kernel
        const Vector<Index, 2> min_uv_kernel = (half_kernel_size - uv_img).cwiseMax(Vector<Index, 2>::Zero());
        const Vector<Index, 2> max_uv_kernel = (kernel_size + (img_size - uv_img)).cwiseMin(kernel_size - Vector<Index, 2>::Ones());

        for (Index v_kernel = min_uv_kernel(1); v_kernel <= max_uv_kernel(1); ++v_kernel)
        for (Index u_kernel = min_uv_kernel(0); u_kernel <= max_uv_kernel(0); ++u_kernel)
        {
            const Vector<Index, 2> uv_kernel(u_kernel, v_kernel);
            const Vector<Index, 2> uv_out = uv_kernel + uv_img - half_kernel_size;

            out(uv_out(1), uv_out(0)) += val * kernel(uv_kernel(1), uv_kernel(0));
        }
    });
}
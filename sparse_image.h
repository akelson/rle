#pragma once

#include "leb128.h"
#include "rle_v2.h"
#include <Eigen/Dense>
#include <concepts>
#include <iterator>

template <typename T=float>
requires std::is_arithmetic_v<T>
using ImArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T=float>
requires std::is_arithmetic_v<T>
using ImArrayRef = Eigen::Ref<ImArray<T>>;

inline std::tuple<bool, uint32_t, std::span<const uint8_t>> decode_run(std::span<const uint8_t> buff, bool prev_value)
{
    auto [run_length, encoded_run_length] = codec::leb128::decode_one<uint8_t>(buff);
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
        while (0 == run_length_ && !encoded_runs.empty())
        {
            auto [new_value, decoded_run_length, encoded_run_length] = decode_run(remaining_encoded_runs_, value_);
            value_ = new_value;
            run_length_ = decoded_run_length;
            remaining_encoded_runs_ = std::span<const uint8_t>(encoded_run_length.end(), encoded_runs.end());
        }

        // Skip runs of zero
        //if (0 == value_)
        //{
        //    index += run_length_ - 1;
        //    run_length_ = 0;
        //    ++(*this);
        //}
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
    assert(kernel.cols() % 2 == 1);
    assert(kernel.rows() % 2 == 1);

    using Eigen::Index;
    using Eigen::Vector;
    visit_sparse_image(image, [&](Eigen::Index u, Eigen::Index v, bool val) {
        const Vector<Index, 2> uv(u, v);
        const Vector<Index, 2> half_kernel_size(kernel.cols() / 2, kernel.rows() / 2);
        const Vector<Index, 2> min_uv = (uv - half_kernel_size).cwiseMax(Vector<Index, 2>(0, 0));
        const Vector<Index, 2> max_uv = (uv + half_kernel_size).cwiseMin(Vector<Index, 2>(out.cols() - 1, out.rows() - 1));

        for (Eigen::Index v = min_uv(1); v <= max_uv(1); ++v)
        {
            for (Eigen::Index u = min_uv(0); u <= max_uv(0); ++u)
            {
                out(v, u) += val * kernel(v - min_uv(1), u - min_uv(0));
            }
        }
    });
}
#pragma once

#include "leb128.h"
#include "rle_v2.h"
#include <Eigen/Dense>
#include <concepts>
#include <iterator>

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
    T& operator*() { return value_; }
    const T& operator*() const { return value_; }

    size_t index() const { return index_; }

  private:
    T value_;
    size_t index_;
    size_t num_px_;
    size_t run_length_;
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

    PixelIterator<T> begin()
    {
        return PixelIterator(init_prev_value<bool>(), 0, size(), encoded_runs_);
    }

    PixelIterator<T> end()
    {
        return PixelIterator<T>(size());
    }

    Eigen::Index width() const { return width_; }
    Eigen::Index height() const { return height_; }
    Eigen::Index size() const { return width_ * height_; }

  private:
    std::span<uint8_t> encoded_runs_;
    Eigen::Index width_;
    Eigen::Index height_;
    std::vector<uint8_t> owned_buff_;
};
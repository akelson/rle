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
class PixelRunIterator
{
  public:

    PixelRunIterator(T prev_value, size_t start_index, size_t img_px_remaining, std::span<const uint8_t> encoded_runs) : 
        value_(prev_value),
        start_index_(start_index),
        img_px_remaining_(img_px_remaining),
        run_length_(),
        remaining_encoded_runs_(encoded_runs)
    {
        auto [new_value, decoded_run_length, encoded_run_length] = decode_run(remaining_encoded_runs_, value_);
        value_ = new_value;
        run_length_ = decoded_run_length;
        remaining_encoded_runs_ = std::span<const uint8_t>(encoded_run_length.end(), remaining_encoded_runs_.end());

        // Skip runs of zero
        while (0 == value_) ++(*this);
    }

    // End iterator
    PixelRunIterator() : img_px_remaining_(std::numeric_limits<size_t>::max()) {}
  
    PixelRunIterator& operator++()
    {
        ++start_index_;
        --img_px_remaining_;

        if (1 == img_px_remaining_)
        {
            // Set to end iterator
            *this = PixelRunIterator();
        }
        else if (run_length_)
        {
            --run_length_;
        }
        else
        {
            *this = PixelRunIterator(value_, start_index_, img_px_remaining_, remaining_encoded_runs_);
        }

        return *this;
    }

    bool operator==(const PixelRunIterator& rhs) const { return img_px_remaining_ == rhs.img_px_remaining_; }
    bool operator!=(const PixelRunIterator& rhs) const { return !(*this == rhs); }
    T& operator*() { return value_; }

    size_t index() const { return start_index_; }

  private:
    T value_;
    size_t start_index_;
    size_t img_px_remaining_;
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

    SparseImage(Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> array);

    PixelRunIterator<T> begin()
    {
        return PixelRunIterator(init_prev_value<bool>(), 0, size(), encoded_runs_);
    }

    PixelRunIterator<T> end()
    {
        return PixelRunIterator<T>();
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
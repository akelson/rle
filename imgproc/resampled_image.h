#include <Eigen/Dense>

template <typename T=float>
requires std::is_arithmetic_v<T>
class ResampleImageOpBase
{
  public:
    const Eigen::Index width() const { return width_; }
    const Eigen::Index height() const { return height_; }

  private:
    Eigen::Index width_;
    Eigen::Index height_;
};

enum class InterpolationMethod
{
    NEAREST_NEIGHBOR,
    BILINEAR,
    BICUBIC
};

template <typename T>
Index image_width(cont T& image) { return image.cols(); }

template <typename T>
Index image_height(cont T& image) { return image.rows(); }

template <typename T=float, InterpolationMethod INTERPOLATION_METHOD, typename ImageType>
requires std::is_arithmetic_v<T>
class ResizeImageOp : public ResampledImageBase<T>
{
  public:
    using PixelType = T;

    PixelType operator()(Eigen::Index i, Eigen::Index j) const
    {
        using Eigen::Vector;
        const Vector<float, 2> uv{static_cast<float>(j), static_cast<float>(i)};
        const Vector<float, 2> min_uv_orig{-0.5f, -0.5f};
        const Vector<float, 2> max_uv_orig{image_width(original_image_) - 0.5f, image_height(original_image_) - 0.5f};
        const Vector<float, 2> min_uv{-0.5f, -0.5f};
        const Vector<float, 2> max_uv{width() - 0.5f, height() - 0.5f};
        const Vector<float, 2> uv_orig = uv * (max_uv_orig - min_uv_orig) / (max_uv - min_uv) + min_uv_orig;
        return interpolate(uv_orig, original_image_, INTERPOLATION_METHOD);
    } 

  private:
    ImageType original_image_;
};
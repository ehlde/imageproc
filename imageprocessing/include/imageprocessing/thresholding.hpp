#pragma once

#include <opencv2/core.hpp>

namespace thresholding
{
constexpr auto MIN_VALUE_U8 = 0U;
constexpr auto MAX_VALUE_U8 = 255U;
constexpr auto NIBLACK_K = -0.17;

enum class ThresholdType : int
{
  OPENCV_GAUSSIAN = 0,
  NIBLACK_NAIVE,
  NIBLACK_INTEGRAL,
};

cv::Mat adaptiveThresholding(
    const cv::Mat& img,
    const ThresholdType type = ThresholdType::OPENCV_GAUSSIAN);

cv::Mat niblackNaive(const cv::Mat& paddedImg,
                     const int blockSize,
                     const double k,
                     const bool invert = false);

cv::Mat niblackIntegral(const cv::Mat& paddedImg,
                        const int blockSize,
                        const double k,
                        const bool invert = false);
}  // namespace thresholding
#pragma once

#include <opencv2/core.hpp>

namespace thresholding
{
constexpr int MIN_VALUE_U8 = 0;
constexpr auto MAX_VALUE_U8 = 255;
enum class ThresholdType : int
{
  OPENCV_GAUSSIAN = 0,
  NIBLACK_NAIVE,
  NIBLACK_INTEGRAL_IMG,
};

cv::Mat adaptive_thresholding(
    const cv::Mat& img, ThresholdType type = ThresholdType::OPENCV_GAUSSIAN);
}  // namespace thresholding
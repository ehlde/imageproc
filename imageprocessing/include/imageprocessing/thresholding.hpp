#pragma once

#include <opencv2/core.hpp>

namespace thresholding
{

cv::Mat adaptive_thresholding(const cv::Mat& img);
}  // namespace thresholding
#pragma once

#include <opencv4/opencv2/core.hpp>

namespace highlights
{
cv::Mat detectHighlights(const cv::Mat& src);
}  // namespace highlights
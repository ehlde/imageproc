#pragma once

#include <opencv2/core.hpp>

namespace highlights
{
cv::Mat detectHighlights(const cv::Mat& src);
}  // namespace highlights
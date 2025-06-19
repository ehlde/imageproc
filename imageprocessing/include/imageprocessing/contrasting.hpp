#pragma once

#include <opencv2/core.hpp>

namespace contrasting
{

cv::Mat histogramEqualization(const cv::Mat& img);

}  // namespace contrasting
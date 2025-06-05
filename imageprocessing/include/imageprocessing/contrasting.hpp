#pragma once

#include <opencv4/opencv2/core.hpp>

namespace contrasting
{

cv::Mat histogramEqualization(const cv::Mat& img);

}  // namespace contrasting
#pragma once

#include <opencv4/opencv2/opencv.hpp>

namespace gemm
{
cv::Mat performGemmCv(const cv::Mat& A, const cv::Mat& B);
}  // namespace gemm
#pragma once

#include <opencv2/core.hpp>

namespace gemm
{
cv::Mat performGemmCv(const cv::Mat& A, const cv::Mat& B);
}  // namespace gemm
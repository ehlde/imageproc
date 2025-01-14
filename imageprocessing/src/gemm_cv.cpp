#include "imageprocessing/gemm_cv.hpp"

namespace gemm
{
cv::Mat performGemmCv(const cv::Mat& A, const cv::Mat& B)
{
  cv::Mat C;
  cv::gemm(A, B, 1.0, cv::Mat(), 0.0, C);
  return C;
}
}  // namespace gemm
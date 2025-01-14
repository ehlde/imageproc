#include "imageprocessing/highlights.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace highlights
{
cv::Mat detectHighlights(const cv::Mat& src)
{
  auto blurred = cv::Mat{};
  cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);

  // Calculate mean and stddev.
  auto mean = cv::Scalar{};
  auto stddev = cv::Scalar{};
  cv::meanStdDev(blurred, mean, stddev);

  // Set threshold as mean + 2*stddev.
  constexpr auto NUM_STDDEV = 2.0;
  const auto threshold = mean[0] + NUM_STDDEV * stddev[0];

  // Create mask for bright pixels.
  auto mask = cv::Mat{};
  cv::threshold(src, mask, threshold, 255, cv::THRESH_BINARY);

  return mask;
}
}  // namespace highlights
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

cv::Mat detectHighlightsTopHat(const cv::Mat& img)
{
  // Apply top-hat transform.
  auto tophat = cv::Mat{};
  cv::morphologyEx(img,
                   tophat,
                   cv::MORPH_TOPHAT,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));

  // Threshold to get binary mask.
  auto mask = cv::Mat{};
  cv::threshold(tophat, mask, 255, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  return mask;
}
}  // namespace highlights
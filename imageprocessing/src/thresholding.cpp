#include "imageprocessing/thresholding.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imageprocessing/highlights.hpp"

namespace
{
auto calculateOptimalC(const cv::Mat& img) -> double
{
  // Get initial threshold using Otsu.
  auto otsu = cv::Mat{};
  const auto otsu_threshold = cv::threshold(img, otsu, 0, 255, cv::THRESH_OTSU);

  // Calculate local contrast.
  auto local_contrast = cv::Mat{};
  cv::Laplacian(img, local_contrast, CV_64F);

  // Get mean local contrast.
  const auto mean_contrast = cv::mean(cv::abs(local_contrast))[0];

  // Adjust C based on contrast - higher contrast needs lower C.
  const auto optimal_c = otsu_threshold / (mean_contrast + 1.0);

  return std::clamp(optimal_c, 2.0, 50.0);
}
}  // namespace

namespace thresholding
{
cv::Mat adaptive_thresholding(const cv::Mat& img)
{
  const auto height = img.rows;
  const auto blockSize = height % 2 == 0 ? height - 1 : height;
  constexpr auto MAX_VALUE = 255;
  const auto subtractionFactor = std::clamp(calculateOptimalC(img), 10.0, 20.0);
  auto result = cv::Mat{};
  cv::adaptiveThreshold(img,
                        result,
                        255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV,
                        blockSize,
                        subtractionFactor);

  // Perform morphological closing to remove noise.
  const auto kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);

  // Global thresholding.
  auto global = cv::Mat{};
  const auto threshold = cv::threshold(img, global, 0, 255, cv::THRESH_OTSU);

  // Invert global threshold.
  cv::bitwise_not(global, global);

  cv::imshow("Original", img);
  cv::imshow("Global Thresholding", global);
  cv::imshow("Adaptive Thresholding", result);
  cv::waitKey(0);
  return result;
}
}  // namespace thresholding
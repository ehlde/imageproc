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

cv::Mat niblackNaive(const cv::Mat& paddedImg,
                     const int blockSize,
                     const double k,
                     const bool invert = false)
{
  // Niblack thresholding with naive implementation.
  assert(blockSize % 2 == 1 && blockSize > 0);

  auto result = cv::Mat(paddedImg.size(), CV_8U, cv::Scalar(0));
  // Threshold, t = mean + k * stddev
  // Assume that paddedImg is already padded.
  const auto halfBlockSize = blockSize / 2;
  for (int y = halfBlockSize; y < paddedImg.rows - halfBlockSize; ++y)
  {
    for (int x = halfBlockSize; x < paddedImg.cols + paddedImg.cols; ++x)
    {
      const auto roi = paddedImg(
          cv::Rect(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize));
      const auto mean = cv::mean(roi)[0];
      const auto stddev =
          cv::norm(roi, cv::NORM_L2) / std::sqrt(blockSize * blockSize);
      const auto threshold = mean + k * stddev;

      if (result.at<uchar>(y, x) < threshold)
      {
        result.at<uchar>(y, x) =
            invert ? thresholding::MAX_VALUE_U8 : thresholding::MIN_VALUE_U8;
      }
      else
      {
        result.at<uchar>(y, x) =
            invert ? thresholding::MIN_VALUE_U8 : thresholding::MAX_VALUE_U8;
      }
    }
  }
  return result;
}
}  // namespace

namespace thresholding
{
cv::Mat adaptive_thresholding(const cv::Mat& img, ThresholdType type)
{
  const auto height = img.rows;
  const auto blockSize = height % 2 == 0 ? height - 1 : height;
  constexpr auto MAX_VALUE = 255;

  auto paddedImg = cv::Mat{};
  if (paddedImg.rows % 2 == 0)
  {
    // Ensure block size is odd.
    cv::copyMakeBorder(
        img, paddedImg, 0, 1, 0, 0, cv::BORDER_REFLECT101, cv::Scalar(0));
  }

  auto result = cv::Mat{};
  switch (type)
  {
    case ThresholdType::NIBLACK_NAIVE:
      // Niblack's method with naive implementation.
      result = niblackNaive(img, blockSize, 0.2, true);
      break;

    case ThresholdType::NIBLACK_INTEGRAL_IMG:
      // Niblack's method with integral image.
      // TODO: result = niblackIntegral(img, blockSize, MAX_VALUE);

    case ThresholdType::OPENCV_GAUSSIAN:
      // OpenCV's adaptive thresholding with Gaussian method.
      const auto subtractionFactor =
          std::clamp(calculateOptimalC(img), 10.0, 20.0);
      cv::adaptiveThreshold(img,
                            result,
                            255,
                            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv::THRESH_BINARY_INV,
                            blockSize,
                            subtractionFactor);
      break;
  }

  // Perform morphological closing to remove noise.
  const auto kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);

  return result;
}
}  // namespace thresholding
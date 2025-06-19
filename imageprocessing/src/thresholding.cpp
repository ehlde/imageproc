#include "imageprocessing/thresholding.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imageprocessing/highlights.hpp"
#include "imageprocessing/integral_image.hpp"

namespace thresholding
{
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

cv::Mat niblackNaive(const cv::Mat& paddedImg,
                     const int halfBlockSize,
                     const double k,
                     const bool invert)
{
  // Niblack thresholding with naive implementation.
  assert(halfBlockSize % 2 == 1 && halfBlockSize > 0);
  const auto blockSize = halfBlockSize * 2;

  auto result = cv::Mat(paddedImg.size(), CV_8U, cv::Scalar(0));
  // Threshold, T = mean + k * stddev
  for (auto y = halfBlockSize; y < paddedImg.rows - halfBlockSize; ++y)
  {
    for (auto x = halfBlockSize; x < paddedImg.cols - halfBlockSize; ++x)
    {
      const auto roi = paddedImg(
          cv::Rect(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize));
      cv::Scalar mean, stdDev;
      cv::meanStdDev(roi, mean, stdDev);
      const auto threshold =
          mean[0] + k * stdDev[0];  // Grayscale image, single channel.
      const auto pixelValue = paddedImg.at<uchar>(y, x);

      if (pixelValue < threshold)
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
  return result(cv::Rect(halfBlockSize,
                         halfBlockSize,
                         paddedImg.cols - blockSize,
                         paddedImg.rows - blockSize));
}

cv::Mat niblackIntegral(const cv::Mat& paddedImg,
                        const int halfBlockSize,
                        const double k,
                        const bool invert)
{
  // Niblack thresholding with integral image.
  assert(halfBlockSize % 2 == 1 && halfBlockSize > 0);
  const auto blockSize = halfBlockSize * 2;
  const auto area = blockSize * blockSize;

  const auto [integralImg, sqrIntegralImg] =
      imageprocessing::integralImageTwoOrders(paddedImg);

  auto result = cv::Mat(paddedImg.size(), CV_8U, cv::Scalar(0));
  for (int y = halfBlockSize; y < paddedImg.rows - halfBlockSize; ++y)
  {
    for (int x = halfBlockSize; x < paddedImg.cols - halfBlockSize; ++x)
    {
      // Calculate the sum of pixels in the block.
      const auto sum =
          integralImg.at<int>(y + halfBlockSize, x + halfBlockSize) -
          integralImg.at<int>(y - halfBlockSize, x + halfBlockSize) -
          integralImg.at<int>(y + halfBlockSize, x - halfBlockSize) +
          integralImg.at<int>(y - halfBlockSize, x - halfBlockSize);

      // Calculate the sum of squared pixels in the block.
      const auto sqrSum =
          sqrIntegralImg.at<int>(y + halfBlockSize, x + halfBlockSize) -
          sqrIntegralImg.at<int>(y - halfBlockSize, x + halfBlockSize) -
          sqrIntegralImg.at<int>(y + halfBlockSize, x - halfBlockSize) +
          sqrIntegralImg.at<int>(y - halfBlockSize, x - halfBlockSize);

      // Calculate mean and standard deviation.
      const auto mean = static_cast<double>(sum) / area;
      const auto variance = static_cast<double>(sqrSum) / area - mean * mean;
      const auto stddev = std::sqrt(variance);

      // Calculate threshold.
      const auto threshold = mean + k * stddev;
      // Apply threshold.
      if (paddedImg.at<uchar>(y, x) < threshold)
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
  return result(cv::Rect(halfBlockSize,
                         halfBlockSize,
                         paddedImg.cols - blockSize,
                         paddedImg.rows - blockSize));
}

cv::Mat adaptiveThresholding(const cv::Mat& img, const ThresholdType type)
{
  const auto blockHeight = img.rows / 4;
  const auto halfBlockSize =
      blockHeight % 2 == 0 ? (blockHeight - 1) / 2 : blockHeight / 2;
  const auto BORDER_REFLECT_VAL = halfBlockSize;

  // Create padded image.
  switch (type)
  {
    case ThresholdType::NIBLACK_NAIVE:
    case ThresholdType::NIBLACK_INTEGRAL:
    case ThresholdType::OPENCV_GAUSSIAN:
      break;
  }
}
}  // namespace thresholding
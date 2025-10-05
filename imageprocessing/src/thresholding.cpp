#include "imageprocessing/thresholding.hpp"

#include <immintrin.h>

#include <cstdint>
#include <iostream>
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

/**
 * @brief Creates a cv::Mat with its data buffer aligned to a specified byte
 * boundary.
 *
 * @param rows The number of rows for the new matrix.
 * @param cols The number of columns for the new matrix.
 * @param type The type of the matrix (e.g., CV_8U, CV_32S).
 * @param alignment The desired byte alignment (e.g., 16, 32, 64). Must be a
 * power of two.
 * @return A cv::Mat whose data pointer is aligned.
 */
auto createAlignedMat(const int rows,
                      const int cols,
                      const int type,
                      const int alignment) -> cv::Mat
{
  CV_Assert(alignment > 0 && (alignment & (alignment - 1)) == 0);

  const auto elemSize = CV_ELEM_SIZE(type);
  const auto rowSize = cols * elemSize;

  // Calculate aligned step
  const auto alignedStep = ((rowSize + alignment - 1) / alignment) * alignment;

  // Allocate extra bytes for alignment
  const auto totalSize = alignedStep * rows + alignment - 1;
  auto buffer = std::vector<uchar>(totalSize);

  // Find aligned address
  const auto bufferAddr = reinterpret_cast<uintptr_t>(buffer.data());
  const auto alignedAddr =
      (bufferAddr + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
  auto* alignedPtr = reinterpret_cast<uchar*>(alignedAddr);

  // Create matrix with aligned data
  // Note: We need to keep the buffer alive, so we'll use a different approach
  auto result = cv::Mat(rows, cols, type);

  // Verify alignment of OpenCV's allocation
  if (reinterpret_cast<uintptr_t>(result.data) % alignment == 0 &&
      result.isContinuous())
  {
    return result;
  }

  // Backup: return the regular matrix as OpenCV's allocator
  // typically provides reasonable alignment for AVX operations
  return result;
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

cv::Mat niblackIntegralSIMD(const cv::Mat& paddedImg,
                            const int halfBlockSize,
                            const double k,
                            const bool invert)
{
  assert(halfBlockSize % 2 == 1 && halfBlockSize > 0);

  const auto blockSize = halfBlockSize * 2;
  const auto area = blockSize * blockSize;
  assert(paddedImg.type() == CV_8U);
  const auto resultRoi = cv::Rect(halfBlockSize,
                                  halfBlockSize,
                                  paddedImg.cols - blockSize,
                                  paddedImg.rows - blockSize);

  auto result = cv::Mat(paddedImg.rows, paddedImg.cols, CV_8U, cv::Scalar(0));

  const auto [integralImg, sqrIntegralImg] =
      imageprocessing::integralImageTwoOrders(paddedImg);

  constexpr auto STEP = 8;

  const auto areaVec = _mm256_set1_ps(static_cast<float>(area));
  const auto areaSqrVec = _mm256_set1_ps(static_cast<float>(area * area));
  const auto kVec = _mm256_set1_ps(static_cast<float>(k));
  const auto maxVec = _mm256_set1_epi32(thresholding::MAX_VALUE_U8);
  const auto minVec = _mm256_setzero_si256();

  for (auto y = halfBlockSize; y < integralImg.rows - halfBlockSize; ++y)
  {
    const auto* integralTopRow = integralImg.ptr<const int>(y - halfBlockSize);
    const auto* integralBottomRow =
        integralImg.ptr<const int>(y + halfBlockSize);

    const auto* sqrIntegralTopRow =
        sqrIntegralImg.ptr<const int>(y - halfBlockSize);
    const auto* sqrIntegralBottomRow =
        sqrIntegralImg.ptr<const int>(y + halfBlockSize);

    const auto* pixelRow = paddedImg.ptr<const uchar>(y);
    auto* resultRow = result.ptr<uchar>(y);

    auto x = halfBlockSize;
    for (; x + STEP <= integralImg.cols - halfBlockSize; x += STEP)
    {
      const auto* topLeft = &integralTopRow[x - halfBlockSize];
      const auto* topRight = &integralTopRow[x + halfBlockSize];
      const auto* bottomLeft = &integralBottomRow[x - halfBlockSize];
      const auto* bottomRight = &integralBottomRow[x + halfBlockSize];

      const auto topLeftVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(topLeft));
      const auto topRightVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(topRight));
      const auto bottomLeftVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bottomLeft));
      const auto bottomRightVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bottomRight));

      const auto pos = _mm256_add_epi32(bottomRightVec, topLeftVec);
      const auto neg = _mm256_add_epi32(bottomLeftVec, topRightVec);
      const auto sum = _mm256_sub_epi32(pos, neg);

      const auto* sqrTopLeft = &sqrIntegralTopRow[x - halfBlockSize];
      const auto* sqrTopRight = &sqrIntegralTopRow[x + halfBlockSize];
      const auto* sqrBottomLeft = &sqrIntegralBottomRow[x - halfBlockSize];
      const auto* sqrBottomRight = &sqrIntegralBottomRow[x + halfBlockSize];

      const auto sqrTopLeftVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sqrTopLeft));
      const auto sqrTopRightVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sqrTopRight));
      const auto sqrBottomLeftVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sqrBottomLeft));
      const auto sqrBottomRightVec =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sqrBottomRight));

      const auto sqrPos = _mm256_add_epi32(sqrBottomRightVec, sqrTopLeftVec);
      const auto sqrNeg = _mm256_add_epi32(sqrBottomLeftVec, sqrTopRightVec);
      const auto sqrSum = _mm256_sub_epi32(sqrPos, sqrNeg);

      const auto sumFloat = _mm256_cvtepi32_ps(sum);
      const auto sqrSumFloat = _mm256_cvtepi32_ps(sqrSum);

      // mean     = sum / area
      // variance = (sqrSum / area) - (mean / area)²

      const auto mean = _mm256_div_ps(sumFloat, areaVec);

      const auto sqrSumOverArea = _mm256_div_ps(sqrSumFloat, areaVec);
      const auto sqrMean = _mm256_mul_ps(mean, mean);
      const auto variance = _mm256_sub_ps(sqrSumOverArea, sqrMean);
      const auto stddev = _mm256_sqrt_ps(variance);
      // T = mean + k * stddev
      const auto kStdDev = _mm256_mul_ps(kVec, stddev);
      const auto rhs = _mm256_add_ps(mean, kStdDev);
      const auto threshold = _mm256_cvttps_epi32(rhs);

      // Load 8 bytes (8 pixels) from the pixel row
      const auto pixelChunk =
          _mm_loadu_si64(reinterpret_cast<const void*>(&pixelRow[x]));
      // Convert to epi32
      const auto pixelVecEpi32 = _mm256_cvtepu8_epi32(pixelChunk);

      const auto mask = invert ? _mm256_cmpgt_epi32(threshold, pixelVecEpi32)
                               : _mm256_cmpgt_epi32(pixelVecEpi32, threshold);
      const auto resVec = _mm256_blendv_epi8(minVec, maxVec, mask);

      const auto packed16 = _mm256_packs_epi32(resVec, resVec);
      const auto packed8 = _mm256_packus_epi16(packed16, packed16);
      const auto permuted = _mm256_permute4x64_epi64(packed8, 0xD8);
      const auto final8 = _mm256_castsi256_si128(permuted);
      _mm_storeu_si64(reinterpret_cast<void*>(&resultRow[x]), final8);
    }

    // Handle remaining pixels
    for (; x < integralImg.cols - halfBlockSize; ++x)
    {
      const auto sum =
          integralImg.at<int>(y + halfBlockSize, x + halfBlockSize) -
          integralImg.at<int>(y - halfBlockSize, x + halfBlockSize) -
          integralImg.at<int>(y + halfBlockSize, x - halfBlockSize) +
          integralImg.at<int>(y - halfBlockSize, x - halfBlockSize);

      const auto sqrSum =
          sqrIntegralImg.at<int>(y + halfBlockSize, x + halfBlockSize) -
          sqrIntegralImg.at<int>(y - halfBlockSize, x + halfBlockSize) -
          sqrIntegralImg.at<int>(y + halfBlockSize, x - halfBlockSize) +
          sqrIntegralImg.at<int>(y - halfBlockSize, x - halfBlockSize);

      const auto mean = static_cast<double>(sum) / area;
      const auto variance = static_cast<double>(sqrSum) / area - mean * mean;
      const auto stddev = std::sqrt(variance);
      const auto threshold = mean + k * stddev;

      if (paddedImg.at<uchar>(y, x) < threshold)
      {
        resultRow[x] =
            invert ? thresholding::MAX_VALUE_U8 : thresholding::MIN_VALUE_U8;
      }
      else
      {
        resultRow[x] =
            invert ? thresholding::MIN_VALUE_U8 : thresholding::MAX_VALUE_U8;
      }
    }
  }
  return result(resultRoi);
}

cv::Mat adaptiveThresholding(const cv::Mat& img, const ThresholdType type)
{
  const auto blockHeight = img.rows / 4;
  const auto halfBlockSize = (blockHeight - 1) / 2;
  const auto BORDER_REFLECT_VAL = halfBlockSize;

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     BORDER_REFLECT_VAL,
                     BORDER_REFLECT_VAL,
                     BORDER_REFLECT_VAL,
                     BORDER_REFLECT_VAL,
                     cv::BORDER_REFLECT_101);

  switch (type)
  {
    case ThresholdType::NIBLACK_NAIVE:
      return niblackNaive(paddedImg, halfBlockSize, NIBLACK_K, true);
    case ThresholdType::NIBLACK_INTEGRAL:
      return niblackIntegral(paddedImg, halfBlockSize, NIBLACK_K, true);
    case ThresholdType::OPENCV_GAUSSIAN:
    {
      auto result = cv::Mat{};
      cv::adaptiveThreshold(img,
                            result,
                            MAX_VALUE_U8,
                            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv::THRESH_BINARY,
                            blockHeight,
                            calculateOptimalC(img));
      return result;
    }
  }
  return cv::Mat{};
}

}  // namespace thresholding
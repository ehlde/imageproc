#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "imageprocessing/thresholding.hpp"

namespace thresholding
{
namespace
{
constexpr auto TOLERANCE = 1e-6;

auto createPaddedImage(const int width,
                       const int height,
                       const int halfBlockSize) -> cv::Mat
{
  auto img = cv::Mat(height, width, CV_8U);
  cv::randu(img, cv::Scalar(0), cv::Scalar(255));

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     cv::BORDER_REFLECT_101);
  return paddedImg;
}

auto createGradientImage(const int width,
                         const int height,
                         const int halfBlockSize) -> cv::Mat
{
  auto img = cv::Mat(height, width, CV_8U);
  for (auto y = 0; y < height; ++y)
  {
    for (auto x = 0; x < width; ++x)
    {
      img.at<uchar>(y, x) = static_cast<uchar>((x * 255) / width);
    }
  }

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     cv::BORDER_REFLECT_101);
  return paddedImg;
}

auto createCheckerboardImage(const int width,
                             const int height,
                             const int checkerSize,
                             const int halfBlockSize) -> cv::Mat
{
  auto img = cv::Mat(height, width, CV_8U);
  for (auto y = 0; y < height; ++y)
  {
    for (auto x = 0; x < width; ++x)
    {
      const auto checkerX = (x / checkerSize) % 2;
      const auto checkerY = (y / checkerSize) % 2;
      img.at<uchar>(y, x) = (checkerX ^ checkerY) ? 255 : 0;
    }
  }

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     halfBlockSize,
                     cv::BORDER_REFLECT_101);
  return paddedImg;
}
}  // namespace

class NiblackIntegralTest : public ::testing::Test
{
 protected:
  static constexpr auto K_VALUE = -0.2;
  static constexpr auto HALF_BLOCK_SIZE = 7;
};

TEST_F(NiblackIntegralTest, BasicFunctionality)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);

  // Check that result contains only binary values
  auto minVal = 0.0;
  auto maxVal = 0.0;
  cv::minMaxLoc(result, &minVal, &maxVal);
  EXPECT_TRUE(minVal == MIN_VALUE_U8 || minVal == MAX_VALUE_U8);
  EXPECT_TRUE(maxVal == MIN_VALUE_U8 || maxVal == MAX_VALUE_U8);
}

// TEST_F(NiblackIntegralTest, CompareWithNaiveImplementation)
// {
//   const auto paddedImg = createPaddedImage(128, 128, HALF_BLOCK_SIZE);

//   const auto integralResult =
//       niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);
//   const auto naiveResult =
//       niblackNaive(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

//   EXPECT_EQ(integralResult.size(), naiveResult.size());

//   // Check that results match
//   auto diff = cv::Mat{};
//   cv::absdiff(integralResult, naiveResult, diff);
//   const auto maxDiff = cv::norm(diff, cv::NORM_INF);

//   // Allow small differences due to floating point precision
//   EXPECT_LE(maxDiff, 1.0);
// }

TEST_F(NiblackIntegralTest, InvertFlagFalse)
{
  const auto paddedImg = createGradientImage(64, 64, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  // With gradient and invert=false, left side should be mostly black (0)
  // and right side should be mostly white (255)
  const auto leftRegion = result(cv::Rect(0, 0, result.cols / 4, result.rows));
  const auto rightRegion =
      result(cv::Rect(3 * result.cols / 4, 0, result.cols / 4, result.rows));

  const auto leftMean = cv::mean(leftRegion)[0];
  const auto rightMean = cv::mean(rightRegion)[0];

  EXPECT_LT(leftMean, rightMean);
}

TEST_F(NiblackIntegralTest, InvertFlagTrue)
{
  const auto paddedImg = createGradientImage(64, 64, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, true);

  // With gradient and invert=true, behavior should be inverted
  const auto leftRegion = result(cv::Rect(0, 0, result.cols / 4, result.rows));
  const auto rightRegion =
      result(cv::Rect(3 * result.cols / 4, 0, result.cols / 4, result.rows));

  const auto leftMean = cv::mean(leftRegion)[0];
  const auto rightMean = cv::mean(rightRegion)[0];

  EXPECT_GT(leftMean, rightMean);
}

TEST_F(NiblackIntegralTest, InvertFlagsAreOpposite)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto resultNormal =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);
  const auto resultInverted =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, true);

  EXPECT_EQ(resultNormal.size(), resultInverted.size());

  // Check that inverted result is the opposite of normal result
  for (auto y = 0; y < resultNormal.rows; ++y)
  {
    for (auto x = 0; x < resultNormal.cols; ++x)
    {
      const auto normalVal = resultNormal.at<uchar>(y, x);
      const auto invertedVal = resultInverted.at<uchar>(y, x);

      if (normalVal == MAX_VALUE_U8)
      {
        EXPECT_EQ(invertedVal, MIN_VALUE_U8);
      }
      else
      {
        EXPECT_EQ(invertedVal, MAX_VALUE_U8);
      }
    }
  }
}

TEST_F(NiblackIntegralTest, DifferentBlockSizes)
{
  for (const auto halfBlockSize : {1, 3, 5, 7, 9, 11, 13, 15})
  {
    const auto paddedTestImg = createPaddedImage(128, 128, halfBlockSize);

    EXPECT_NO_THROW({
      const auto result =
          niblackIntegral(paddedTestImg, halfBlockSize, K_VALUE, false);
      EXPECT_GT(result.rows, 0);
      EXPECT_GT(result.cols, 0);

      // Verify the result has the expected dimensions
      const auto blockSize = halfBlockSize * 2;
      const auto expectedWidth = paddedTestImg.cols - blockSize;
      const auto expectedHeight = paddedTestImg.rows - blockSize;
      EXPECT_EQ(result.cols, expectedWidth);
      EXPECT_EQ(result.rows, expectedHeight);
    });
  }
}

TEST_F(NiblackIntegralTest, DifferentKValues)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto resultK1 =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, -0.5, false);
  const auto resultK2 = niblackIntegral(paddedImg, HALF_BLOCK_SIZE, 0.0, false);
  const auto resultK3 = niblackIntegral(paddedImg, HALF_BLOCK_SIZE, 0.5, false);

  // Results should be different for different k values
  auto diff1 = cv::Mat{};
  auto diff2 = cv::Mat{};
  cv::absdiff(resultK1, resultK2, diff1);
  cv::absdiff(resultK2, resultK3, diff2);

  EXPECT_GT(cv::countNonZero(diff1), 0);
  EXPECT_GT(cv::countNonZero(diff2), 0);
}

TEST_F(NiblackIntegralTest, UniformImage)
{
  constexpr auto UNIFORM_VALUE = 128;
  auto img = cv::Mat(64, 64, CV_8U, cv::Scalar(UNIFORM_VALUE));

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     cv::BORDER_REFLECT_101);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  // For uniform image with stddev=0: threshold = mean + k*0 = mean
  // All pixels equal mean, so result should be uniform
  auto minVal = 0.0;
  auto maxVal = 0.0;
  cv::minMaxLoc(result, &minVal, &maxVal);

  EXPECT_EQ(minVal, maxVal);
  EXPECT_TRUE(minVal == MIN_VALUE_U8 || minVal == MAX_VALUE_U8);
}

TEST_F(NiblackIntegralTest, LargeImage)
{
  const auto paddedImg = createPaddedImage(512, 512, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);
}

TEST_F(NiblackIntegralTest, SmallImage)
{
  constexpr auto SMALL_HALF_BLOCK_SIZE = 1;
  const auto paddedImg = createPaddedImage(16, 16, SMALL_HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, SMALL_HALF_BLOCK_SIZE, K_VALUE, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);
}

TEST_F(NiblackIntegralTest, ResultRoiSize)
{
  constexpr auto WIDTH = 100;
  constexpr auto HEIGHT = 80;
  const auto paddedImg = createPaddedImage(WIDTH, HEIGHT, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  // Result should exclude the padded borders
  const auto blockSize = HALF_BLOCK_SIZE * 2;
  const auto expectedWidth = paddedImg.cols - blockSize;
  const auto expectedHeight = paddedImg.rows - blockSize;

  EXPECT_EQ(result.cols, expectedWidth);
  EXPECT_EQ(result.rows, expectedHeight);
}

TEST_F(NiblackIntegralTest, CheckerboardPattern)
{
  constexpr auto SIZE = 64;
  constexpr auto CHECKER_SIZE = 8;
  const auto paddedImg =
      createCheckerboardImage(SIZE, SIZE, CHECKER_SIZE, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  // Result should maintain some pattern structure
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);

  // Check that we have both black and white regions
  auto minVal = 0.0;
  auto maxVal = 0.0;
  cv::minMaxLoc(result, &minVal, &maxVal);
  EXPECT_EQ(minVal, MIN_VALUE_U8);
  EXPECT_EQ(maxVal, MAX_VALUE_U8);
}

TEST_F(NiblackIntegralTest, EdgeCasesZeroStdDev)
{
  // Create an image with regions of constant value
  constexpr auto SIZE = 64;
  auto img = cv::Mat(SIZE, SIZE, CV_8U, cv::Scalar(100));

  // Add a single different pixel
  img.at<uchar>(SIZE / 2, SIZE / 2) = 200;

  auto paddedImg = cv::Mat{};
  cv::copyMakeBorder(img,
                     paddedImg,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     HALF_BLOCK_SIZE,
                     cv::BORDER_REFLECT_101);

  EXPECT_NO_THROW({
    const auto result =
        niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);
    EXPECT_GT(result.rows, 0);
    EXPECT_GT(result.cols, 0);
  });
}

TEST_F(NiblackIntegralTest, NegativeKValue)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto result = niblackIntegral(paddedImg, HALF_BLOCK_SIZE, -0.5, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);

  // Verify binary output
  auto minVal = 0.0;
  auto maxVal = 0.0;
  cv::minMaxLoc(result, &minVal, &maxVal);
  EXPECT_TRUE(minVal == MIN_VALUE_U8 || minVal == MAX_VALUE_U8);
  EXPECT_TRUE(maxVal == MIN_VALUE_U8 || maxVal == MAX_VALUE_U8);
}

TEST_F(NiblackIntegralTest, PositiveKValue)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto result = niblackIntegral(paddedImg, HALF_BLOCK_SIZE, 0.5, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);

  // Verify binary output
  auto minVal = 0.0;
  auto maxVal = 0.0;
  cv::minMaxLoc(result, &minVal, &maxVal);
  EXPECT_TRUE(minVal == MIN_VALUE_U8 || minVal == MAX_VALUE_U8);
  EXPECT_TRUE(maxVal == MIN_VALUE_U8 || maxVal == MAX_VALUE_U8);
}

TEST_F(NiblackIntegralTest, RectangularImage)
{
  const auto paddedImg = createPaddedImage(120, 80, HALF_BLOCK_SIZE);

  const auto result =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  EXPECT_EQ(result.type(), CV_8U);
  EXPECT_GT(result.rows, 0);
  EXPECT_GT(result.cols, 0);

  const auto blockSize = HALF_BLOCK_SIZE * 2;
  EXPECT_EQ(result.cols, paddedImg.cols - blockSize);
  EXPECT_EQ(result.rows, paddedImg.rows - blockSize);
}

TEST_F(NiblackIntegralTest, ConsistencyAcrossRuns)
{
  const auto paddedImg = createPaddedImage(64, 64, HALF_BLOCK_SIZE);

  const auto result1 =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);
  const auto result2 =
      niblackIntegral(paddedImg, HALF_BLOCK_SIZE, K_VALUE, false);

  // Results should be identical for the same input
  auto diff = cv::Mat{};
  cv::absdiff(result1, result2, diff);
  const auto maxDiff = cv::norm(diff, cv::NORM_INF);

  EXPECT_EQ(maxDiff, 0.0);
}

}  // namespace thresholding

#include <immintrin.h>

#include <cassert>
#include <imageprocessing/integral_image.hpp>
#include <opencv2/core.hpp>

namespace imageprocessing
{
cv::Mat integralImage(const cv::Mat& img)
{
  assert(img.type() == CV_8U &&
         "Input image must be of type CV_8U (single channel, 8-bit unsigned "
         "integer).");
  // Compute the integral image of the input image.
  cv::Mat integralImg = cv::Mat::zeros(img.size(), CV_32S);

  for (int i = 0; i < img.rows; ++i)
  {
    for (int j = 0; j < img.cols; ++j)
    {
      // Calculate the sum of pixels in the rectangle from (0,0) to (i,j).
      const auto sum = img.at<uchar>(i, j) +
                       (i > 0 ? integralImg.at<int>(i - 1, j) : 0) +
                       (j > 0 ? integralImg.at<int>(i, j - 1) : 0) -
                       (i > 0 && j > 0 ? integralImg.at<int>(i - 1, j - 1) : 0);
      integralImg.at<int>(i, j) = sum;
    }
  }

  return integralImg;
}

std::pair<cv::Mat, cv::Mat> integralImageTwoOrders(const cv::Mat& img)
{
  assert(img.type() == CV_8U &&
         "Input image must be of type CV_8U (single channel, 8-bit unsigned "
         "integer).");
  // Compute the integral image of the input image with two orders.
  cv::Mat integralImg = cv::Mat::zeros(img.size(), CV_32S);
  cv::Mat sqrIntegralImg = cv::Mat::zeros(img.size(), CV_32S);

  for (int y = 0; y < img.rows; ++y)
  {
    for (int x = 0; x < img.cols; ++x)
    {
      const auto sum = img.at<uchar>(y, x) +
                       (y > 0 ? integralImg.at<int>(y - 1, x) : 0) +
                       (x > 0 ? integralImg.at<int>(y, x - 1) : 0) -
                       (y > 0 && x > 0 ? integralImg.at<int>(y - 1, x - 1) : 0);
      integralImg.at<int>(y, x) = sum;

      const auto sqrSum =
          img.at<uchar>(y, x) * img.at<uchar>(y, x) +
          (y > 0 ? sqrIntegralImg.at<int>(y - 1, x) : 0) +
          (x > 0 ? sqrIntegralImg.at<int>(y, x - 1) : 0) -
          (y > 0 && x > 0 ? sqrIntegralImg.at<int>(y - 1, x - 1) : 0);
      sqrIntegralImg.at<int>(y, x) = sqrSum;
    }
  }

  return {integralImg, sqrIntegralImg};
}

std::pair<cv::Mat, cv::Mat> integralImageTwoOrders_SIMD(const cv::Mat& img)
{
  assert(false);  // NOT IMPLEMENTED.
  assert(img.type() == CV_8U &&
         "Input image must be of type CV_8U (single channel, 8-bit unsigned "
         "integer).");
  cv::Mat integralImg = cv::Mat::zeros(img.size(), CV_32S);
  cv::Mat sqrIntegralImg = cv::Mat::zeros(img.size(), CV_32S);

  // Create padded image with an additional row and column of zeros.
  // This is necessary to avoid boundary checks during the integral image.
  cv::Mat paddedImg = cv::Mat::zeros(img.rows + 1, img.cols + 1, CV_8U);
  img.copyTo(paddedImg(cv::Rect(1, 1, img.cols, img.rows)));

  // TODO
  constexpr auto VEC_SIZE = 4;
  for (int y = 1; y < paddedImg.rows; y += VEC_SIZE)
  {
    for (int x = 1; x < paddedImg.cols; x += VEC_SIZE)
    {
      // I(x,y) = I(x-1,y) + I(x,y-1) - I(x-1,y-1) + img(x,y)

      // Calculate the sum of squared pixels in the rectangle from (0,0) to
      // (y,x).
      const auto sqrSum =
          paddedImg.at<uchar>(y, x) * paddedImg.at<uchar>(y, x) +
          sqrIntegralImg.at<int>(y - 1, x) + sqrIntegralImg.at<int>(y, x - 1) -
          sqrIntegralImg.at<int>(y - 1, x - 1);
      sqrIntegralImg.at<int>(y - 1, x - 1) = sqrSum;
    }
  }
  return {integralImg, sqrIntegralImg};
}
}  // namespace imageprocessing

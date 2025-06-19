#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/photo.hpp>

#include "imageprocessing/highlights.hpp"

namespace imageprocessing
{
namespace reflection_removal
{
struct HighlightMask
{
  cv::Mat mask;
  std::vector<cv::Point2f> centers;
};

struct Ray
{
  cv::Point2f start;
  float angle;
};

cv::Mat getEdgeImage(const cv::Mat& img)
{
  // Image preprocessing should already be done.
  cv::Mat edges;
  cv::Laplacian(img, edges, CV_8U, 5);
  return edges;
}

HighlightMask getHighlightsAndCenters(const cv::Mat& img)
{
  // Image preprocessing should already be done.

  // Get highlights and their respective centers.
  auto highlights = highlights::detectHighlights(img);
  const auto structuringElement =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::dilate(highlights, highlights, structuringElement, cv::Point(-1, -1), 3);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(
      highlights, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Get center of each contour.
  std::vector<cv::Point2f> contourCenters;
  for (const auto& contour : contours)
  {
    cv::Moments moments = cv::moments(contour);
    contourCenters.push_back(
        cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00));
  }

  return {highlights, contourCenters};
}

auto removeReflectionsFromCentralEllipse(const cv::Mat& img,
                                         const cv::Point2f& pointInEllipse)
    -> cv::Mat
{
  auto denoised = cv::Mat{};
  cv::fastNlMeansDenoising(img, denoised, 10.0f, 7, 21);

  auto blurred = cv::Mat{};
  cv::GaussianBlur(denoised, blurred, cv::Size(5, 5), 0);

  const auto edges = getEdgeImage(blurred);

  const auto highlightsAndCenters = getHighlightsAndCenters(blurred);
  for (const auto& center : highlightsAndCenters.centers)
  {
    cv::circle(blurred, center, 1, cv::Scalar(1), -1);
  }

  // Get contours of highlights.

  cv::imshow("Input", img);
  cv::imshow("Blurred", blurred);
  cv::imshow("Edges", edges);
  cv::waitKey(0);
  return cv::Mat();
}
}  // namespace reflection_removal
}  // namespace imageprocessing
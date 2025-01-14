#include "imageprocessing/contrasting.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace contrasting
{
cv::Mat histogramEqualization(const cv::Mat& img)
{
  cv::imshow("Original", img);
  cv::Mat equalizedHist;
  cv::equalizeHist(img, equalizedHist);
  return equalizedHist;
}
}  // namespace contrasting
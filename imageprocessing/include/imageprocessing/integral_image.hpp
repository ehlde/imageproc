
#include <opencv2/core.hpp>

namespace imageprocessing
{
cv::Mat integralImage(const cv::Mat& img);

std::pair<cv::Mat, cv::Mat> integralImageTwoOrders(const cv::Mat& img);
}  // namespace imageprocessing
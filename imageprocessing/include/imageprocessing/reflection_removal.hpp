#include <opencv2/core.hpp>

namespace imageprocessing
{
namespace reflection_removal
{
auto removeReflectionsFromCentralEllipse(const cv::Mat& img,
                                         const cv::Point2f& pointInEllipse)
    -> cv::Mat;
}  // namespace reflection_removal
}  // namespace imageprocessing
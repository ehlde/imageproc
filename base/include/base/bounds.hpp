#pragma once

#include <opencv2/core.hpp>

namespace base
{
bool isInBounds(const cv::Size& imgSize, const cv::Point& point)
{
  return point.x >= 0 && point.x < imgSize.width && point.y >= 0 &&
         point.y < imgSize.height;
}
}  // namespace base

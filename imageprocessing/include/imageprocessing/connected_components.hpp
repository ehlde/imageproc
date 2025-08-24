#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "base/types.hpp"

namespace imageprocessing
{
cv::Mat connectedEightSlow(const cv::Mat& img);
cv::Mat connectedEightFaster(const cv::Mat& img);
}
#include "imageprocessing/connected_components.hpp"

#include <queue>

#include "base/bounds.hpp"
#include "base/types.hpp"

namespace imageprocessing
{
constexpr auto RECT_SIZE = 3;
constexpr auto STEP = 1;
namespace
{
void getNeighborsRecursive(const cv::Mat& img,
                           const cv::Rect& rect,
                           const int currentLabel,
                           cv::Mat& inOutLabels)
{
  for (int x = rect.x; x < rect.x + rect.width; ++x)
  {
    for (int y = rect.y; y < rect.y + rect.height; ++y)
    {
      if (x == rect.x + rect.width / 2 && y == rect.y + rect.height / 2)
      {
        continue;  // Skip the center pixel
      }
      if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
      {
        if (img.at<uchar>(y, x) > 0 && inOutLabels.at<uchar>(y, x) == 0)
        {
          inOutLabels.at<uchar>(y, x) = currentLabel;
          getNeighborsRecursive(
              img,
              cv::Rect(x - STEP, y - STEP, RECT_SIZE, RECT_SIZE),
              currentLabel,
              inOutLabels);
        }
      }
    }
  }
}
}  // namespace

// This is very slow by design.
// - Loops through the image in a way that causes cache misses.
// - Recursively explores neighbors, leading to deep call stacks.
// - Generally ugly.
cv::Mat connectedEightSlow(const cv::Mat& img)
{
  assert(img.type() == CV_8U &&
         "Input image must be of type CV_8U (single channel, 8-bit unsigned "
         "integer).");

  cv::Mat labels = cv::Mat::zeros(img.size(), CV_8U);

  auto currentLabel = 1;
  for (int x = 0; x < img.cols; ++x)
  {
    for (int y = 0; y < img.rows; ++y)
    {
      if (labels.at<uchar>(y, x) == 0 && img.at<uchar>(y, x) > 0)
      {
        labels.at<uchar>(y, x) = currentLabel;

        getNeighborsRecursive(
            img,
            cv::Rect2i(x - STEP, y - STEP, RECT_SIZE, RECT_SIZE),
            currentLabel,
            labels);
        currentLabel++;
      }
    }
  }

  return labels;
}

cv::Mat connectedEightFaster(const cv::Mat& img)
{
  assert(img.type() == CV_8U &&
         "Input image must be of type CV_8U (single channel, 8-bit unsigned "
         "integer).");

  cv::Mat labels = cv::Mat::zeros(img.size(), CV_8U);
  auto currentLabel = 1;

  for (int y = 0; y < img.rows; ++y)
  {
    for (int x = 0; x < img.cols; ++x)
    {
      if (labels.at<uchar>(y, x) == 0 && img.at<uchar>(y, x) > 0)
      {
        labels.at<uchar>(y, x) = currentLabel;

        // Use a queue for breadth-first search (BFS)
        auto queue = std::queue<cv::Point>({cv::Point(x, y)});

        while (!queue.empty())
        {
          const auto point = queue.front();
          queue.pop();

          // Add all unlabeled neighbors to the queue
          for (int dy = -1; dy <= 1; ++dy)
          {
            for (int dx = -1; dx <= 1; ++dx)
            {
              if (dx == 0 && dy == 0)
              {
                continue;  // Skip the center pixel
              }
              if (base::isInBounds(img.size(),
                                   cv::Point(point.x + dx, point.y + dy)) &&
                  img.at<uchar>(point.y + dy, point.x + dx) > 0 &&
                  labels.at<uchar>(point.y + dy, point.x + dx) == 0)
              {
                labels.at<uchar>(point.y + dy, point.x + dx) = currentLabel;
                queue.push(cv::Point(point.x + dx, point.y + dy));
              }
            }
          }
        }
        currentLabel++;
      }
    }
  }
  return labels;
}
}  // namespace imageprocessing

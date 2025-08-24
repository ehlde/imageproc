#pragma once

#include <opencv2/core.hpp>

namespace base
{
struct Point : cv::Point2i
{
  Point() : cv::Point2i(0, 0) {}
  Point(int x, int y) : cv::Point2i(x, y) {}
};

class Pixel
{
 public:
  Pixel() : value(0), position(0, 0) {}
  Pixel(uint8_t v, const Point& p) : value(v), position(p) {}
  uint8_t getValue() const
  {
    return value;
  }
  void setValue(uint8_t v)
  {
    value = v;
  }
  void setPosition(const Point& p)
  {
    position = p;
  }
  Point getPosition() const
  {
    return position;
  }
  bool sanityCheck() const
  {
    return value > 0 && position.x >= 0 && position.y >= 0;
  }

 private:
  uint8_t value;
  Point position;
};
}  // namespace base
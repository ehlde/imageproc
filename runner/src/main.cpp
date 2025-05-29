#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <vector>

#include "imageprocessing/contrasting.hpp"
#include "imageprocessing/highlights.hpp"
#include "imageprocessing/reflection_removal.hpp"
#include "imageprocessing/thresholding.hpp"

using Images = std::vector<cv::Mat>;

std::vector<std::string> get_file_paths(const std::string& folder,
                                        const std::string& extension)
{
  std::vector<std::string> file_paths;
  for (const auto& entry : std::filesystem::directory_iterator(folder))
  {
    const auto& path = entry.path();
    if (path.extension() == extension)
    {
      file_paths.push_back(path.string());
    }
  }
  return file_paths;
}

Images loadImages(const std::vector<std::string>& file_paths)
{
  Images images;
  for (const auto& file : file_paths)
  {
    const auto src = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
      std::cerr << "Error: Image not found\n";
      return {};
    }
    images.push_back(src);
  }
  return images;
}

Images performThresholding(const Images& images)
{
  Images results{};
  for (const auto& image : images)
  {
    results.push_back(thresholding::adaptive_thresholding(image));
  }
  return results;
}

void performAutoContrast(std::vector<std::string>& file_paths)
{
  for (const auto& file : file_paths)
  {
    const auto src = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
      std::cerr << "Error: Image not found\n";
      return;
    }

    auto dst = contrasting::histogramEqualization(src);
  }
}

void perform_reflection_removal(const Images& images,
                                std::vector<cv::Point2f> points)
{
  if (images.size() != points.size())
  {
    std::cerr << "Error: Number of images and points do not match\n";
    return;
  }

  for (size_t i = 0; i < images.size(); ++i)
  {
    const auto& image = images[i];
    const auto& point = points[i];

    const auto result = imageprocessing::reflection_removal::
        remove_reflections_from_central_ellipse(image, point);

    // cv::imshow("Input", image);
    // cv::imshow("Result", result);
    // cv::waitKey(0);
  }
}

std::vector<std::vector<cv::Point>> getPointsAtEdgeOfHighlight(
    const cv::Mat& image)
{
  std::vector<cv::Point> borderPoints;
  const auto highlights = highlights::detectHighlights(image);
  const auto structuringElement =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::dilate(highlights, highlights, structuringElement, cv::Point(-1, -1), 3);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(
      highlights, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

std::vector<cv::Point> getSurroundingPixelsNotInMask(const cv::Mat& mask,
                                                     const cv::Point& point)
{
  constexpr auto KERNEL_SIZE = 3;
  std::vector<cv::Point> surroundingPixels;
  for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; ++i)
  {
    for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; ++j)
    {
      const auto x = point.x + i;
      const auto y = point.y + j;
      if (x >= 0 && x < mask.rows && y >= 0 && y < mask.cols)
      {
        if (mask.at<uchar>(x, y) == 0)
        {
          surroundingPixels.emplace_back(x, y);
        }
      }
    }
  }
  return surroundingPixels;
}

cv::Mat cvInpainting(const cv::Mat& src)
{
  cv::Mat dst = src.clone();

  while (true)
  {
    auto contours = getPointsAtEdgeOfHighlight(src);
    if (contours.empty())
    {
      break;
    }
    for (const auto& contour : contours)
    {
      if (contour.empty())
      {
        continue;
      }
      for (const auto& point : contour)
      {
        const auto surroundingPixels =
            getSurroundingPixelsNotInMask(dst, point);
        // Get average of surrounding pixels.
        auto averageColor = 0.0;
        std::ranges::for_each(
            surroundingPixels,
            [&averageColor, &dst](const auto& pixel)
            { averageColor += dst.at<uchar>(pixel.x, pixel.y); });
        averageColor /= surroundingPixels.size();
        dst.at<uchar>(point.x, point.y) = averageColor;
      }
    }
  }

  cv::imshow("Inpainting", dst);
  cv::waitKey(0);
  return dst;
}

int main(const int argc, const char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <folder_path> <extension>\n";
    return 1;
  }

  const auto folder_path = std::string(argv[1]);
  const auto extension = std::string(argv[2]);

  std::vector<std::string> files{};
  try
  {
    files = get_file_paths(folder_path, extension);
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  const auto images = loadImages(files);
  // const std::vector<cv::Point2f> points{cv::Point2f(93, 23),
  //                                       cv::Point2f(119, 49),
  //                                       cv::Point2f(123, 54),
  //                                       cv::Point2f(94, 34)};
  // // perform_reflection_removal(images, points);
  // for (const auto& image : images)
  // {
  //   cvInpainting(image);
  // }

  const auto result = performThresholding(images);
  for (auto idx = 0; idx < result.size(); ++idx)
  {
    const auto& image = result[idx];
    const auto file_name = std::filesystem::path(files[idx]).filename();
    cv::imshow(file_name.string(), image);
    if (!result.empty())
    {
      cv::imshow("Thresholded Image", result[0]);
    }
    cv::waitKey(0);
  }

  return 0;
}

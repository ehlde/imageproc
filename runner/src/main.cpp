#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

#include "imageprocessing/contrasting.hpp"
#include "imageprocessing/highlights.hpp"
#include "imageprocessing/reflection_removal.hpp"
#include "imageprocessing/thresholding.hpp"

using Images = std::vector<cv::Mat>;

std::vector<std::string> getFilePaths(const std::string& folder,
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

Images loadVideo(const std::string& file_path, const int frame_count = 0)
{
  Images images;
  cv::VideoCapture cap(file_path);
  if (!cap.isOpened())
  {
    std::cerr << "Error: Could not open video file\n";
    return {};
  }

  cv::Mat frame;
  while (cap.read(frame) && images.size() < frame_count)
  {
    // Convert to grayscale if the frame is not empty
    if (frame.channels() > 1)
    {
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }
    // If the frame is empty, break the loop
    if (frame.empty())
    {
      break;
    }
    // Push the frame to the images vector
    images.push_back(frame);
  }
  return images;
}

Images performThresholding(const Images& images)
{
  Images results{};
  for (const auto& image : images)
  {
    results.push_back(thresholding::adaptive_thresholding(
        image, thresholding::ThresholdType::OPENCV_GAUSSIAN));
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

  const auto folderPath = std::string(argv[1]);
  const auto extension = std::string(argv[2]);

  auto files = std::vector<std::string>{};
  try
  {
    files = getFilePaths(folderPath, extension);
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  const auto images = loadImages(files);
  // const auto images = loadVideo(files[0], 10);

  const auto result = performThresholding(images);
  for (auto idx = 0; idx < result.size(); ++idx)
  {
    if (!result.empty() && idx < std::ssize(images))
    {
      // Combine original and thresholded images side by side.
      auto combined = cv::Mat{};
      cv::hconcat(images[idx], result[idx], combined);

      cv::imshow("Original | Thresholded", combined);
      cv::waitKey(0);
    }
  }

  return 0;
}

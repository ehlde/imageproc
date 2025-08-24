#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/videoio.hpp>
#include <span>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "imageprocessing/connected_components.hpp"
#include "imageprocessing/contrasting.hpp"
#include "imageprocessing/highlights.hpp"
#include "imageprocessing/reflection_removal.hpp"
#include "imageprocessing/thresholding.hpp"

using Images = std::vector<cv::Mat>;
using ImagePairs = std::vector<std::pair<cv::Mat, cv::Mat>>;

namespace
{
auto getFilePaths(const std::string& folderPath, const std::string& extension)
    -> std::vector<std::string>
{
  auto filePaths = std::vector<std::string>{};
  for (const auto& entry : std::filesystem::directory_iterator(folderPath))
  {
    const auto& path = entry.path();
    if (path.extension() == extension)
    {
      filePaths.push_back(path.string());
    }
  }
  return filePaths;
}

auto loadImages(const std::vector<std::string>& filePaths) -> Images
{
  auto images = Images{};
  for (const auto& file : filePaths)
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

auto loadVideo(const std::string& filePath, const int frameCount = 0) -> Images
{
  auto images = Images{};
  cv::VideoCapture cap(filePath);
  if (!cap.isOpened())
  {
    std::cerr << "Error: Could not open video file\n";
    return {};
  }

  auto frame = cv::Mat{};
  while (cap.read(frame) && images.size() < frameCount)
  {
    if (frame.channels() > 1)
    {
      cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }
    if (frame.empty())
    {
      break;
    }
    images.push_back(frame);
  }
  return images;
}

auto performThresholding(const Images& images) -> ImagePairs
{
  auto results = ImagePairs{};
  constexpr auto BLOCK_SIZE = 15;
  constexpr auto K = -0.17;
  for (const auto& image : images)
  {
    const auto naive = thresholding::adaptiveThresholding(
        image, thresholding::ThresholdType::NIBLACK_NAIVE);
    const auto integral = thresholding::adaptiveThresholding(
        image, thresholding::ThresholdType::NIBLACK_INTEGRAL);
    results.push_back({naive, integral});
  }
  return results;
}

auto performAutoContrast(std::vector<std::string>& filePaths) -> void
{
  for (const auto& file : filePaths)
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
        removeReflectionsFromCentralEllipse(image, point);
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

static void BM_NiblackNaive(benchmark::State& state,
                            const cv::Mat& paddedImg,
                            const int halfBlockSize,
                            const double kValue,
                            const bool invert)
{
  // Benchmark NiblackNaive function.
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(
        thresholding::niblackNaive(paddedImg, halfBlockSize, kValue, invert));
  }
}

static void BM_NiblackIntegral(benchmark::State& state,
                               const cv::Mat& paddedImg,
                               const int halfBlockSize,
                               const double kValue,
                               const bool invert)
{
  // Benchmark NiblackIntegral function.
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(thresholding::niblackIntegral(
        paddedImg, halfBlockSize, kValue, invert));
  }
}

static void BM_ConnectedEightSlow(benchmark::State& state,
                                  const cv::Mat& testImg)
{
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(imageprocessing::connectedEightSlow(testImg));
  }
}

static void BM_ConnectedEightFaster(benchmark::State& state,
                                    const cv::Mat& testImg)
{
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(imageprocessing::connectedEightFaster(testImg));
  }
}

void performAdaptiveThresholding(const Images& images) {}

void performConnectedComponents(const Images& images)
{
  for (const auto& image : images)
  {
    const auto resultSlow = imageprocessing::connectedEightSlow(image);
    const auto resultFaster = imageprocessing::connectedEightFaster(image);

    auto maskSlow = cv::Mat{};
    cv::threshold(resultSlow, maskSlow, 0, 255, cv::THRESH_BINARY);

    auto maskFaster = cv::Mat{};
    cv::threshold(resultFaster, maskFaster, 0, 255, cv::THRESH_BINARY);

    // Compare the masks
    cv::Mat maskComparison;
    cv::compare(maskSlow, maskFaster, maskComparison, cv::CMP_EQ);
    assert(cv::countNonZero(maskComparison) == maskComparison.total());

    // Show all results concatenated.
    cv::Mat allResults;
    cv::hconcat(image, maskSlow, allResults);
    cv::hconcat(allResults, maskFaster, allResults);
    cv::imshow("All Results", allResults);

    cv::waitKey(0);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  const auto args = std::span{argv, static_cast<size_t>(argc)};

  // Expecting at least program name, folder path, and extension.
  if (args.size() < 3)
  {
    std::cout << "Usage: " << args[0]
              << " <folder_path> <extension> [options...]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --benchmark : Run performance benchmarks for selected "
                 "algorithms that have benchmarks implemented."
              << std::endl;
    std::cout << "Algorithms:" << std::endl;
    std::cout << "  --at  : Run adaptive thresholding." << std::endl;
    std::cout << "  --cc  : Run connected components." << std::endl;
    std::cout << "  --rr  : Run reflection removal." << std::endl;
    std::cout << "  --ip  : Run inpainting on highlights." << std::endl;
    std::cout << "  --ii  : Run integral image." << std::endl;
    std::cout << "  --ac  : Run auto contrast." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "  --h   : Show this help message." << std::endl;
    return 1;
  }

  const auto folderPath = std::string{args[1]};
  const auto extension = std::string{args[2]};

  // Lambda to extract optional arguments into a hash set for fast lookups.
  auto getOptionalArgsAsSet = [](const std::span<char*>& allArgs)
      -> std::unordered_set<std::string_view>
  {
    if (allArgs.size() <= 3)
    {
      return {};
    }
    // Create a view of the optional arguments, skipping the first 3.
    auto optionalArgsSpan = allArgs.subspan(3);
    return {optionalArgsSpan.begin(), optionalArgsSpan.end()};
  };

  const auto optionalArgs = getOptionalArgsAsSet(args);

  auto files = std::vector<std::string>{};
  try
  {
    files = getFilePaths(folderPath, extension);
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  if (files.empty())
  {
    std::cerr << "Error: No images with extension '" << extension
              << "' found in folder '" << folderPath << "'." << std::endl;
    return 1;
  }

  const auto images = loadImages(files);
  if (images.empty())
  {
    std::cerr << "Error: Failed to load any images." << std::endl;
    return 1;
  }

  if (optionalArgs.contains("--at"))
  {
    if (optionalArgs.contains("--benchmark"))
    {
      std::cout << "Running benchmarks..." << std::endl;
      benchmark::Initialize(&argc, argv);

      const auto& img = images[0];

      constexpr auto DEFAULT_INVERT = true;
      const auto blockHeight = img.rows / 4;
      const auto halfBlockSize =
          blockHeight % 2 == 0 ? (blockHeight - 1) / 2 : blockHeight / 2;
      const auto BORDER_REFLECT_VAL = halfBlockSize;

      auto paddedImg = cv::Mat{};
      cv::copyMakeBorder(img,
                         paddedImg,
                         BORDER_REFLECT_VAL,
                         BORDER_REFLECT_VAL,
                         BORDER_REFLECT_VAL,
                         BORDER_REFLECT_VAL,
                         cv::BORDER_REFLECT_101);

      benchmark::RegisterBenchmark("BM_NiblackNaive",
                                   BM_NiblackNaive,
                                   paddedImg,
                                   halfBlockSize,
                                   thresholding::NIBLACK_K,
                                   DEFAULT_INVERT);
      benchmark::RegisterBenchmark("BM_NiblackIntegral",
                                   BM_NiblackIntegral,
                                   paddedImg,
                                   halfBlockSize,
                                   thresholding::NIBLACK_K,
                                   DEFAULT_INVERT);

      benchmark::RunSpecifiedBenchmarks();
      benchmark::Shutdown();
    }
    else
    {
      // Default behavior: visualize the results.
      const auto result = performThresholding(images);
      for (const auto& [naive, integral] : result)
      {
        assert(!naive.empty() && !integral.empty() &&
               naive.size() == integral.size());
        auto combined = cv::Mat{};
        cv::hconcat(naive, integral, combined);
        cv::imshow("Naive | Integral", combined);
        cv::waitKey(0);
      }
    }
  }

  if (optionalArgs.contains("--cc"))
  {
    if (optionalArgs.contains("--benchmark"))
    {
      std::cout << "Running connected components benchmarks..." << std::endl;
      benchmark::Initialize(&argc, argv);
      benchmark::RegisterBenchmark(
          "BM_ConnectedEightSlow", BM_ConnectedEightSlow, images[1]);
      benchmark::RegisterBenchmark(
          "BM_ConnectedEightFaster", BM_ConnectedEightFaster, images[1]);
      benchmark::RunSpecifiedBenchmarks();
      benchmark::Shutdown();
    }
    else
    {
      std::cout << "Running connected components..." << std::endl;
      performConnectedComponents(images);
    }
  }
  return 0;
}

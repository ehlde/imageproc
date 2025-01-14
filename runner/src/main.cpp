#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "imageprocessing/contrasting.hpp"
#include "imageprocessing/thresholding.hpp"

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

std::vector<cv::Mat> loadImages(const std::vector<std::string>& file_paths)
{
  std::vector<cv::Mat> images;
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

std::vector<cv::Mat> performThresholding(const std::vector<cv::Mat>& images)
{
  std::vector<cv::Mat> results{};
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
  performThresholding(images);

  return 0;
}
